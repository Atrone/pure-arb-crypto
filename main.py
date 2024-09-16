import time
import ccxt
import pandas as pd
from itertools import combinations
from tenacity import retry, wait_fixed, stop_after_attempt
import logging
from decimal import Decimal, getcontext

# Configure logging at the beginning of your script
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Set precision for Decimal operations to handle small and large numbers accurately
getcontext().prec = 28

# Define a simple in-memory cache for exchange rates
exchange_rate_cache = {}
CACHE_TTL = 60  # Time-to-live for cache entries in seconds

def get_usd_to_quote_rate(exchange: ccxt.Exchange, quote_currency: str) -> Decimal:
    """
    Fetch the USD to Quote Currency conversion rate.
    
    :param exchange: CCXT exchange instance
    :param quote_currency: The quote currency (e.g., 'EUR')
    :return: Conversion rate as Decimal
    """
    cache_key = f"USD/{quote_currency}"
    current_time = time.time()
    
    # Check if the rate is cached and still valid
    if cache_key in exchange_rate_cache:
        rate, timestamp = exchange_rate_cache[cache_key]
        if current_time - timestamp < CACHE_TTL:
            logging.info(f"Using cached exchange rate for {cache_key}: {rate}")
            return rate
    
    # Attempt to fetch the conversion rate
    try:
        # Some exchanges might use 'EUR/USD' instead of 'USD/EUR'
        try:
            ticker = exchange.fetch_ticker(f"USD/{quote_currency}")
            rate = Decimal(str(ticker['last']))
            logging.info(f"Fetched rate {rate} for {cache_key} from USD/{quote_currency}")
        except ccxt.BadSymbol:
            # Try the inverse pair
            ticker = exchange.fetch_ticker(f"{quote_currency}/USD")
            inverse_rate = Decimal(str(ticker['last']))
            rate = Decimal('1') / inverse_rate
            logging.info(f"Fetched rate {rate} for {cache_key} from {quote_currency}/USD (inverse)")
        
        # Cache the fetched rate
        exchange_rate_cache[cache_key] = (rate, current_time)
        return rate
    except Exception as e:
        logging.error(f"Error fetching USD to {quote_currency} rate: {e}")
        return Decimal('0.0')  # Return zero if unable to fetch rate

def estimate_slippage(exchange, symbol: str, trade_size_usd: float, side: str = 'buy') -> float:
    """
    Estimate the slippage for a given trade on an exchange.

    :param exchange: CCXT exchange instance
    :param symbol: Trading symbol (e.g., 'ETH/EUR' or 'BTC/USD')
    :param trade_size_usd: Trade size in USD
    :param side: 'buy' or 'sell'
    :return: Slippage as a decimal (e.g., 0.001 for 0.1%)
    """
    try:
        order_book = exchange.fetch_order_book(symbol)
    except Exception as e:
        logging.error(f"Error fetching order book for {symbol} on {exchange.id}: {e}")
        return 0.0  # Default to 0% slippage if unable to fetch

    # Parse symbol to get base and quote currencies
    try:
        base_currency, quote_currency = symbol.split('/')
    except Exception as e:
        logging.error(f"Error parsing symbol '{symbol}': {e}")
        return 0.0

    # Determine market price based on trade side
    if side == 'buy':
        orders = order_book.get('asks', [])
        if not order_book['asks']:
            logging.warning(f"No asks available for {symbol}; cannot perform buy slippage estimation.")
            return 0.0
        market_price = Decimal(str(order_book['asks'][0][0]))
    elif side == 'sell':
        orders = order_book.get('bids', [])
        if not order_book['bids']:
            logging.warning(f"No bids available for {symbol}; cannot perform sell slippage estimation.")
            return 0.0
        market_price = Decimal(str(order_book['bids'][0][0]))
    else:
        raise ValueError("side must be either 'buy' or 'sell'")


    if not orders:
        logging.warning("No orders available to execute against.")
        return 0.0  # No orders to execute against

    # Convert trade_size from USD to base currency
    if quote_currency != 'USD':
        # Initialize a CCXT exchange that supports USD to Quote Currency conversion
        # You can choose any reliable exchange; here, we'll use Kraken as an example
        converter_exchange = ccxt.kraken()
        converter_exchange.load_markets()
        usd_to_quote_rate = get_usd_to_quote_rate(converter_exchange, quote_currency)
        
        if usd_to_quote_rate == Decimal('0.0'):
            logging.error(f"Unable to convert USD to {quote_currency}; cannot estimate slippage.")
            return 0.0
        
        trade_size_quote = Decimal(str(trade_size_usd)) * usd_to_quote_rate
    else:
        trade_size_quote = Decimal(str(trade_size_usd))

    trade_size_base = trade_size_quote / market_price if market_price != Decimal('0.0') else Decimal('0.0')
    logging.info(f"Trade Size in {base_currency}: {trade_size_base} {base_currency}")

    # Initialize variables for accumulation
    accumulated = Decimal('0.0')
    cost = Decimal('0.0')

    for order in orders:
        if len(order) < 2:
            logging.warning(f"Unexpected order format: {order}")
            continue  # Skip malformed orders

        price, amount = order[:2]
        price_dec = Decimal(str(price))
        amount_dec = Decimal(str(amount))
        # Determine if entire order can be used or only a portion
        if accumulated + amount_dec < trade_size_base:
            # Use entire order
            accumulated += amount_dec
            cost += price_dec * amount_dec
        else:
            # Use only the needed portion to meet trade_size_base
            needed = trade_size_base - accumulated
            if needed > Decimal('0.0'):
                accumulated += needed
                cost += price_dec * needed
            break  # Trade size met


    if accumulated == Decimal('0.0'):
        logging.warning("No orders matched the trade size.")
        return 0.0  # Avoid division by zero

    # Calculate average price
    if accumulated < trade_size_base:
        # Trade size not fully matched; average price based on accumulated
        average_price = cost / accumulated if accumulated else Decimal('0.0')
    else:
        # Trade size fully matched
        average_price = cost / trade_size_base


    if market_price == Decimal('0.0'):
        logging.warning("Market price is zero.")
        return 0.0  # Avoid division by zero

    # Calculate slippage
    slippage = (average_price - market_price) / market_price

    return float(slippage)

# Configuration
exchanges = ['binance', 'kraken', 'bitfinex']  # Add more exchanges as needed
timeframe = '1m'  # Timeframe for OHLCV data
arbitrage_threshold = 0  # Minimum arbitrage threshold (0.2%)
trade_size = 100  # Example trade size, adjust as needed

# Initialize CCXT exchange instances with API keys if necessary
exchange_instances = {exchange: getattr(ccxt, exchange)() for exchange in exchanges}

# Function to fetch exchange fees (assuming a flat fee structure for simplicity)
def get_exchange_fees(exchange):
    """
    Retrieve the trading fees for a given exchange.
    
    :param exchange: CCXT exchange instance
    :return: Trading fee as a decimal (e.g., 0.001 for 0.1%)
    """
    try:
        fees = exchange.fetch_trading_fees()
        # Assuming 'info' contains 'maker' and 'taker' fees
        maker_fee = fees.get('maker', 0.001)  # Default to 0.1% if not available
        taker_fee = fees.get('taker', 0.001)  # Default to 0.1% if not available
        return maker_fee, taker_fee
    except Exception as e:
        logging.error(f"Error fetching fees for {exchange.id}: {e}")
        return 0.001, 0.001  # Default fees

# Function to fetch supported symbols for each exchange
def get_supported_symbols(exchange):
    try:
        return set(exchange.load_markets().keys())
    except Exception as e:
        logging.error(f"Error fetching markets for {exchange.name}: {e}")
        return set()


exchange_symbols = {name: get_supported_symbols(instance) for name, instance in exchange_instances.items()}

# Identify matching pairs across exchanges
matching_pairs = {}
for (exchange1, symbols1), (exchange2, symbols2) in combinations(exchange_symbols.items(), 2):
    common_symbols = symbols1 & symbols2  # Intersection of symbols
    for symbol in common_symbols:
        if symbol not in matching_pairs:
            matching_pairs[symbol] = []
        matching_pairs[symbol].append((exchange1, exchange2))

# Function to fetch historical data
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def fetch_data(exchange_name, symbol, timeframe, since=None, limit=5):
    exchange = exchange_instances[exchange_name]
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
    return df['close']

# Function to simulate arbitrage
def simulate_arbitrage(symbol, exchanges_pair, trade_size_usd):
    """
    Simulate arbitrage between two exchanges based on the given symbol and trade size in USD.
    
    :param symbol: Trading symbol (e.g., 'ETH/EUR' or 'BTC/USD')
    :param exchanges_pair: Tuple containing (buy_exchange, sell_exchange)
    :param arbitrage_threshold: Minimum arbitrage threshold (e.g., 0.002 for 0.2%)
    :param trade_size_usd: Trade size in USD
    :return: Tuple containing total profit and number of trades executed
    """
    buy_exchange_name, sell_exchange_name = exchanges_pair
    buy_exchange = exchange_instances[buy_exchange_name]
    sell_exchange = exchange_instances[sell_exchange_name]

    # Retrieve fees
    buy_maker_fee, buy_taker_fee = get_exchange_fees(buy_exchange)
    sell_maker_fee, sell_taker_fee = get_exchange_fees(sell_exchange)
    total_fees = buy_taker_fee + sell_taker_fee  # Assuming taker fees for immediate execution

    logging.info(f"Simulating arbitrage for {symbol} between {buy_exchange_name} and {sell_exchange_name}...")

    # Fetch and merge historical data
    data_buy = fetch_data(buy_exchange_name, symbol, timeframe)
    data_sell = fetch_data(sell_exchange_name, symbol, timeframe)
    combined_data = pd.concat([data_buy, data_sell], axis=1, keys=[buy_exchange_name, sell_exchange_name]).dropna()
    combined_data.index = pd.to_datetime(combined_data.index, utc=True)
    combined_data = combined_data.sort_index().dropna()
    profit = 0.0
    num_trades = 0

    for timestamp, prices in combined_data.iterrows():
        buy_price = prices[buy_exchange_name]
        sell_price = prices[sell_exchange_name]
        logging.info(f"Buy price: {buy_price:.5f}")
        logging.info(f"Sell price: {sell_price:.5f}")

        # Estimate slippage for buying and selling
        buy_slippage = estimate_slippage(buy_exchange, symbol, trade_size_usd, side='buy')
        sell_slippage = estimate_slippage(sell_exchange, symbol, trade_size_usd, side='sell')

        logging.info(f"Buy slippage: {buy_slippage * 100:.2f}%")
        logging.info(f"Sell slippage: {sell_slippage * 100:.2f}%")

        # Convert trade sizes for accurate calculations
        # Since trade_size_usd was converted within estimate_slippage,
        # Adjusted buy and sell prices already factor in slippage

        # Calculate adjusted prices based on slippage
        adjusted_buy_price = buy_price * (1 + buy_slippage)
        adjusted_sell_price = sell_price * (1 - sell_slippage)

        # Calculate potential profit before fees
        potential_profit = abs(adjusted_sell_price - adjusted_buy_price)
        logging.info(f"Potential profit (before fees): {potential_profit:.10f}")

        # Calculate fee costs
        # Assuming trade_size_usd is used for both buy and sell
        # Convert trade_size_usd to quote currency for fee calculation
        try:
            base_currency, quote_currency = symbol.split('/')
        except Exception as e:
            logging.error(f"Error parsing symbol '{symbol}': {e}")
            continue  # Skip to next iteration

        fee_cost_buy = Decimal(str(adjusted_buy_price)) * Decimal(str(buy_taker_fee))
        fee_cost_sell = Decimal(str(adjusted_sell_price)) * Decimal(str(sell_taker_fee))
        fee_cost = fee_cost_buy + fee_cost_sell
        logging.info(f"Fee cost: {fee_cost:.10f}")


        # Calculate net profit
        net_profit = Decimal(str(potential_profit)) - fee_cost

        # Determine if net profit meets the arbitrage threshold
        # Threshold is based on the buy price
        try:
            # Initialize a CCXT exchange for USD conversion
            converter_exchange = ccxt.kraken()
            converter_exchange.load_markets()
            usd_to_quote_rate = get_usd_to_quote_rate(converter_exchange, quote_currency)
                
            if usd_to_quote_rate != Decimal('0.0'):
                net_profit_usd = net_profit * usd_to_quote_rate
                logging.info(f"Net Profit in USD: {net_profit_usd:.5f} USD")
                profit += float(net_profit_usd)
                num_trades += 1
                logging.info(f"{timestamp}: Buy on {buy_exchange_name} at {adjusted_buy_price:.5f} {quote_currency} "
                                    f"(slippage: {buy_slippage * 100:.2f}%), Sell on {sell_exchange_name} at {adjusted_sell_price:.5f} {quote_currency} "
                                    f"(slippage: {sell_slippage * 100:.2f}%). Net Profit: {net_profit_usd:.5f} USD")
            else:
                logging.error("USD to quote currency rate is zero; cannot convert net profit to USD.")
        except Exception as e:
            logging.error(f"Error converting net profit to USD: {e}")
            net_profit_usd = Decimal('0.0')

    return profit, num_trades

# Comprehensive report storage
report = {}

# Run the arbitrage simulation for all matching pairs
for symbol, exchange_pairs in matching_pairs.items():
    for exchange_pair in exchange_pairs:
        total_profit, total_trades = simulate_arbitrage(symbol, exchange_pair, trade_size)
        report[(symbol, exchange_pair)] = {'profit': total_profit, 'trades': total_trades}

# Display comprehensive report
logging.info("\nComprehensive Arbitrage Report:")
for (symbol, exchange_pair), result in report.items():
    logging.info(f"Pair: {symbol} | Exchanges: {exchange_pair} | Total Profit: {result['profit']:.2f} | Total Trades: {result['trades']}")

# Summary statistics
overall_profit = sum([result['profit'] for result in report.values()])
overall_trades = sum([result['trades'] for result in report.values()])
logging.info(f"\nOverall Total Profit: {overall_profit:.2f}")
logging.info(f"Overall Total Trades: {overall_trades}")