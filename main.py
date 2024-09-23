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
getcontext().prec = 50  # Increased from 28 to 50

# Define a simple in-memory cache for exchange rates
exchange_rate_cache = {}
CACHE_TTL = 60  # Time-to-live for cache entries in seconds


def get_total_volume(exchange):
    """
    Calculate the total 24h trading volume for an exchange.
    
    :param exchange: Instance of a CCXT exchange
    :return: Total trading volume as float
    """
    VOLUME_THRESHOLD = 2_000_000  # Adjust this value as needed
    try:
        markets = exchange.load_markets()
        total_volume = 0.0
        for symbol in markets:
            ticker = exchange.fetch_ticker(symbol)
            # Use quote volume as a proxy; adjust if needed
            volume = ticker.get('quoteVolume', 0.0)
            if volume:
                total_volume += float(volume)
                if total_volume >= VOLUME_THRESHOLD:
                    logging.info(f"Exchange '{exchange.id}' volume {total_volume} exceeds threshold {VOLUME_THRESHOLD}. Excluding from low liquidity selection.")
                    return float('inf')  # Assign a high volume to exclude this exchange
        return total_volume
    except Exception as e:
        logging.warning(f"Could not fetch volume for {exchange.id}: {e}")
        return float('inf')  # Assign a high volume to exclude problematic exchanges


def select_low_liquidity_exchanges(desired_count=10):
    """
    Select exchanges with the lowest liquidity based on 24h trading volume.
    
    :param desired_count: Number of low liquidity exchanges to select
    :return: List of exchange IDs
    """
    available_exchanges = ccxt.exchanges
    exchange_volumes = {}

    for idx, exchange_id in enumerate(available_exchanges):
        logging.info(f"Processing exchange {idx + 1}/{len(available_exchanges)}: {exchange_id}")
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,  # Respect rate limits
            })
            exchange.load_markets()
            total_volume = get_total_volume(exchange)
            exchange_volumes[exchange_id] = total_volume
            logging.info(f"Exchange: {exchange_id}, Total Volume: {total_volume}")
        except Exception as e:
            logging.warning(f"Failed to process {exchange_id}: {e}")
            exchange_volumes[exchange_id] = float('inf')  # Exclude problematic exchanges

        # Be polite and wait to respect rate limits
        time.sleep(exchange.rateLimit / 1000)

    # Sort exchanges by total_volume in ascending order and select the top 'desired_count'
    sorted_exchanges = sorted(exchange_volumes.items(), key=lambda item: item[1])
    low_liquidity_exchanges = [exchange_id for exchange_id, volume in sorted_exchanges[:desired_count]]

    logging.info(f"Selected Low Liquidity Exchanges: {low_liquidity_exchanges}")
    return low_liquidity_exchanges


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

#bigone
#bequant
#bitcoincom

# Configuration
# Select 10 low liquidity exchanges
exchanges = ['bequant', 'bitcoincom']  #select_low_liquidity_exchanges(desired_count=10)
#'bigone', 
if not exchanges:
    logging.error("No valid exchanges found. Please check the exchange names and your CCXT installation.")
    exit(1)  # Exit the program if no exchanges are loaded

# Initialize CCXT exchange instances with API keys if necessary
exchange_instances = {exchange: getattr(ccxt, exchange)() for exchange in exchanges}
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



def is_spot_market(exchange, symbol):
    try:
        markets = exchange.load_markets()
        if symbol in markets:
            return markets[symbol].get('spot', False)
        return False
    except Exception as e:
        logging.error(f"Error loading markets for {exchange.id}: {e}")
        return False

# Function to fetch historical data
@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def fetch_data(exchange_name, symbol, timeframe, since=None, limit=5):
    exchange = exchange_instances[exchange_name]
    if not is_spot_market(exchange, symbol):
        logging.warning(f"Symbol {symbol} is not a spot market on {exchange_name}. Skipping OHLCV fetch.")
        return None  # Or handle accordingly

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
        return df['close']
    except ccxt.BadRequest as e:
        logging.error(f"BadRequest for symbol {symbol} on exchange {exchange_name}: {e}")
        raise e  # Let tenacity handle the retry
    except Exception as e:
        logging.error(f"Error fetching data for symbol {symbol} on exchange {exchange_name}: {e}")
        raise e
# Function to simulate arbitrage
def simulate_arbitrage(symbol, exchanges_pair, trade_size_usd):
    """
    Simulate arbitrage between two exchanges in both directions based on the given symbol and trade size in USD.
    
    :param symbol: Trading symbol (e.g., 'ETH/EUR' or 'BTC/USD')
    :param exchanges_pair: Tuple containing (exchange1, exchange2)
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
    if data_buy is None or data_sell is None:
        logging.warning(f"Skipping arbitrage simulation for {symbol} between {buy_exchange_name} and {sell_exchange_name} due to missing data.")
        return 0.0, 0
    combined_data = pd.concat([data_buy, data_sell], axis=1, keys=[buy_exchange_name, sell_exchange_name]).dropna()
    combined_data.index = pd.to_datetime(combined_data.index, utc=True)
    combined_data = combined_data.sort_index().dropna()
    profit = 0.0
    num_trades = 0

    for timestamp, prices in combined_data.iterrows():
        buy_price = prices[buy_exchange_name]
        sell_price = prices[sell_exchange_name]
        logging.info(f"Buy price on {buy_exchange_name}: {buy_price:.5f}")
        logging.info(f"Sell price on {sell_exchange_name}: {sell_price:.5f}")

        # Estimate slippage for buying and selling
        buy_slippage = estimate_slippage(buy_exchange, symbol, trade_size_usd, side='buy')
        sell_slippage = estimate_slippage(sell_exchange, symbol, trade_size_usd, side='sell')

        logging.info(f"Buy slippage: {buy_slippage * 100:.2f}%")
        logging.info(f"Sell slippage: {sell_slippage * 100:.2f}%")

        # Direction 1: Buy on Exchange A, Sell on Exchange B
        adjusted_buy_price_A = buy_price * (1 + buy_slippage)
        adjusted_sell_price_A = sell_price * (1 - sell_slippage)

        # Calculate potential profit before fees
        potential_profit_A = Decimal(str(adjusted_sell_price_A)) - Decimal(str(adjusted_buy_price_A))
        logging.info(f"Potential profit (Buy on {buy_exchange_name}, Sell on {sell_exchange_name}): {potential_profit_A:.10f}")

        # Calculate fee costs for Direction 1
        fee_cost_buy_A = Decimal(str(adjusted_buy_price_A)) * Decimal(str(buy_taker_fee))
        fee_cost_sell_A = Decimal(str(adjusted_sell_price_A)) * Decimal(str(sell_taker_fee))
        fee_cost_A = fee_cost_buy_A + fee_cost_sell_A
        logging.info(f"Fee cost Direction 1: {fee_cost_A:.10f}")

        # Calculate net profit for Direction 1
        net_profit_A = potential_profit_A - fee_cost_A

        # Initialize currency variables
        try:
            base_currency, quote_currency = symbol.split('/')
        except Exception as e:
            logging.error(f"Error parsing symbol '{symbol}': {e}")
            continue  # Skip to next iteration

        # Convert net profit to USD if necessary
        if quote_currency != 'USD':
            converter_exchange = ccxt.kraken()
            converter_exchange.load_markets()
            usd_to_quote_rate = get_usd_to_quote_rate(converter_exchange, quote_currency)

            if usd_to_quote_rate != Decimal('0.0'):
                net_profit_usd_A = net_profit_A * usd_to_quote_rate
            else:
                logging.error(f"USD to {quote_currency} rate is zero; cannot convert net profit to USD for Direction 1.")
                net_profit_usd_A = Decimal('0.0')
        else:
            net_profit_usd_A = net_profit_A

        if net_profit_usd_A > Decimal(str(arbitrage_threshold)):
            profit += float(net_profit_usd_A)
            num_trades += 1
            logging.info(
                f"{timestamp}: Direction 1 - Buy on {buy_exchange_name} at {adjusted_buy_price_A:.5f} {quote_currency} "
                f"(slippage: {buy_slippage * 100:.2f}%), Sell on {sell_exchange_name} at {adjusted_sell_price_A:.5f} {quote_currency} "
                f"(slippage: {sell_slippage * 100:.2f}%). Net Profit: {net_profit_usd_A:.5f} USD"
            )
        else:
            logging.info(f"Direction 1 - No profitable arbitrage opportunity. Net Profit: {net_profit_usd_A:.5f} USD")

        # Direction 2: Buy on Exchange B, Sell on Exchange A
        adjusted_buy_price_B = sell_price * (1 + sell_slippage)
        adjusted_sell_price_B = buy_price * (1 - buy_slippage)

        # Calculate potential profit before fees
        potential_profit_B = Decimal(str(adjusted_sell_price_B)) - Decimal(str(adjusted_buy_price_B))
        logging.info(f"Potential profit (Buy on {sell_exchange_name}, Sell on {buy_exchange_name}): {potential_profit_B:.10f}")

        # Calculate fee costs for Direction 2
        fee_cost_buy_B = Decimal(str(adjusted_buy_price_B)) * Decimal(str(sell_taker_fee))
        fee_cost_sell_B = Decimal(str(adjusted_sell_price_B)) * Decimal(str(buy_taker_fee))
        fee_cost_B = fee_cost_buy_B + fee_cost_sell_B
        logging.info(f"Fee cost Direction 2: {fee_cost_B:.10f}")

        # Calculate net profit for Direction 2
        net_profit_B = potential_profit_B - fee_cost_B

        # Convert net profit to USD if necessary
        if quote_currency != 'USD':
            if usd_to_quote_rate != Decimal('0.0'):
                net_profit_usd_B = net_profit_B * usd_to_quote_rate
            else:
                logging.error(f"USD to {quote_currency} rate is zero; cannot convert net profit to USD for Direction 2.")
                net_profit_usd_B = Decimal('0.0')
        else:
            net_profit_usd_B = net_profit_B

        if net_profit_usd_B > Decimal(str(arbitrage_threshold)):
            profit += float(net_profit_usd_B)
            num_trades += 1
            logging.info(
                f"{timestamp}: Direction 2 - Buy on {sell_exchange_name} at {adjusted_buy_price_B:.5f} {quote_currency} "
                f"(slippage: {sell_slippage * 100:.2f}%), Sell on {buy_exchange_name} at {adjusted_sell_price_B:.5f} {quote_currency} "
                f"(slippage: {buy_slippage * 100:.2f}%). Net Profit: {net_profit_usd_B:.5f} USD"
            )
        else:
            logging.info(f"Direction 2 - No profitable arbitrage opportunity. Net Profit: {net_profit_usd_B:.5f} USD")

    return profit, num_trades

# Comprehensive report storage
report = {}

# Run the arbitrage simulation for all matching pairs
for symbol, exchange_pairs in matching_pairs.items():
    for exchange_pair in exchange_pairs:
        total_profit, total_trades = simulate_arbitrage('BCH/DAI', exchange_pair, trade_size)
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


# BTC/DAI
# BTC/

# INFO: # NEGATIVE BECAUSE OF FEE... FEE IS VERY HIGH ON HIGH PRICE CRYPTOS
""" Comprehensive Arbitrage Report:
INFO:Pair: XRP/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5
INFO:Pair: XRP/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XRP/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETH/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ETH/GBP | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETH/GBP | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: QTUM/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: AVAX/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.11 | Total Trades: 
5
INFO:Pair: AVAX/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: AVAX/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: OP/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.03 | Total Trades: 5   
INFO:Pair: DOT/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.03 | Total Trades: 5INFO:Pair: DOT/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: DOT/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: LINK/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.03 | Total Trades: 5INFO:Pair: SOL/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: SUI/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: EOS/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: LTC/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: XRP/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: EOS/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: SOL/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: SOL/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SOL/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ZEC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ZEC/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ZEC/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ICX/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: WIF/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.01 | Total Trades: 5  
INFO:Pair: ARB/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: EOS/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5
INFO:Pair: EOS/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: EOS/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MATIC/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: MATIC/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: MATIC/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: GALA/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: ADA/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BTC/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: BTC/GBP | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BTC/GBP | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: LTC/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: UNI/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: FIL/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ATOM/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.03 | Total Trades: 5INFO:Pair: NEAR/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.01 | Total Trades: 5INFO:Pair: DOGE/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: WBTC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: DOGE/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ALGO/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ETH/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: KSM/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: TRX/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BAT/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: DOT/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: DOT/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: DOT/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETH/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -15.79 | Total Trades: 
5
INFO:Pair: ETH/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ETH/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: REP/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: LPT/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ZRX/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ADA/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ADA/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ADA/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: SNX/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: SAND/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: XRP/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BCH/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.13 | Total Trades: 5 
INFO:Pair: PEPE/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: APT/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.02 | Total Trades: 5  
INFO:Pair: XTZ/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: XTZ/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XTZ/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: GRT/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BCH/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 1.28 | Total Trades: 5 
INFO:Pair: UNI/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: OMG/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: OCEAN/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: MATIC/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: KEEP/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: UNI/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ETH/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: ETH/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETH/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: YFI/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: BCH/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: DOT/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.01 | Total Trades: 5 
INFO:Pair: SOL/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.03 | Total Trades: 5 
INFO:Pair: SOL/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SOL/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BAT/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: XMR/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: XMR/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: XMR/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: LTC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: LTC/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: LTC/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BTC/DAI | Exchanges: ('binance', 'kraken') | Total Profit: -215.24 | Total Trades: 
5
INFO:Pair: APE/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: APE/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: APE/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ATOM/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.01 | Total Trades: 
5
INFO:Pair: ATOM/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ATOM/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SHIB/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ETC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ETC/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETC/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: LINK/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: LINK/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: LINK/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: KNC/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: TRX/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: TRX/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: TRX/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DAI/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BTT/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: LTC/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.27 | Total Trades: 5 
INFO:Pair: ETH/DAI | Exchanges: ('binance', 'kraken') | Total Profit: -19.09 | Total Trades: 5INFO:Pair: MKR/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: XLM/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: XTZ/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: XTZ/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: XTZ/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: RUNE/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BTC/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -499.70 | Total Trades: 5
INFO:Pair: BTC/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: BTC/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XRP/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: GAL/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: DOGE/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: DOGE/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: DOGE/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SHIB/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: SHIB/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SHIB/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ENJ/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: XLM/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: XLM/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XLM/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: LINK/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ATOM/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: POL/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: ENJ/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: OMG/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: OMG/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: OMG/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETH/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -14.89 | Total Trades: 5INFO:Pair: ETH/EUR | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETH/EUR | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DOT/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: STORJ/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: KSM/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: NANO/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BTC/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: SNX/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ALGO/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: ALGO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ALGO/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: USTC/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: GRT/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BTC/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -275.35 | Total Trades: 
5
INFO:Pair: BTC/EUR | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BTC/EUR | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DASH/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: DASH/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: DASH/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SOL/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BAL/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: NANO/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MLN/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: KAVA/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: KAVA/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ALGO/ETH | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MATIC/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: USDC/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.01 | Total Trades: 
5
INFO:Pair: USDC/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: USDC/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: LINK/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DOGE/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 4INFO:Pair: ADA/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ETH/USDC | Exchanges: ('binance', 'kraken') | Total Profit: -19.60 | Total Trades: 
5
INFO:Pair: AVAX/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.10 | Total Trades: 5INFO:Pair: ETC/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ADA/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: ADA/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ANT/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: AAVE/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ENJ/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ATOM/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: DOGE/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5INFO:Pair: DOGE/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: DOGE/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SRM/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: FTM/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: MANA/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 
5
INFO:Pair: ENA/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: DOT/GBP | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: EGLD/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.16 | Total Trades: 5 
INFO:Pair: RENDER/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.02 | Total Trades: 5
INFO:Pair: XMR/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: XMR/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XMR/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: FLOW/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: ADA/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ADA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ADA/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: CRV/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: EOS/USDT | Exchanges: ('binance', 'kraken') | Total Profit: 0.01 | Total Trades: 5 
INFO:Pair: EOS/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: EOS/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SOL/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: ICX/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: ICP/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.04 | Total Trades: 5 
INFO:Pair: LTC/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.53 | Total Trades: 5INFO:Pair: LTC/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: LTC/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETH/JPY | Exchanges: ('binance', 'kraken') | Total Profit: -384982.30 | Total Trades: 5
INFO:Pair: ETH/JPY | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETH/JPY | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: SOL/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.71 | Total Trades: 5 
INFO:Pair: CHZ/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: MANA/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: LINK/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: LINK/AUD | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MINA/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: JASMY/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: TRX/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: TRX/EUR | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: TRX/EUR | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: FIL/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: AAVE/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: COMP/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: GMT/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5  
INFO:Pair: PAXG/BTC | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5INFO:Pair: ANKR/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: MATIC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: MATIC/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: MATIC/BTC | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SC/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0   
INFO:Pair: LUNA/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: QTUM/BTC | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 5 
INFO:Pair: XRP/USDT | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 4INFO:Pair: XRP/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: XRP/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XRP/EUR | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: APE/EUR | Exchanges: ('binance', 'kraken') | Total Profit: 0.00 | Total Trades: 0  
INFO:Pair: BTC/JPY | Exchanges: ('binance', 'kraken') | Total Profit: -7746064.00 | Total Trades: 5
INFO:Pair: BTC/JPY | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BTC/JPY | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETC/ETH | Exchanges: ('binance', 'kraken') | Total Profit: -0.00 | Total Trades: 5 
INFO:Pair: BTC/USDC | Exchanges: ('binance', 'kraken') | Total Profit: -400.28 | Total Trades: 5
INFO:Pair: SNX/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: IOTA/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: BLUR/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: DYM/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: TUSD/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: NEXO/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SPELL/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: NEO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: JUP/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: YFI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: STRK/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: FIL/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ENA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: STG/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FET/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: GBP/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SUI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: GALA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: JASMY/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: EUR/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: WAVES/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: TRX/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: WOO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: XLM/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ETC/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: INJ/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: PEPE/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: FLOKI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SEI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: 1INCH/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ZRO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FTM/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: APT/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: EGLD/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ICP/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: THETA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: VET/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: NOT/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: NEXO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SAND/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: UNI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: PORTAL/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: NEAR/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: AAVE/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SUN/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: POL/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: COMP/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: JST/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: VET/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BONK/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: MEME/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: KAVA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: TIA/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: SUSHI/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: LDO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: CRV/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ARB/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: AXS/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: TON/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: GRT/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: CELO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: AVAX/BTC | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: BTC/TRY | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: WIF/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: TURBO/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: MKR/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FORTH/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: CHZ/USDT | Exchanges: ('binance', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: MKR/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: POL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SEI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ZRX/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: STRK/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ETHW/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: TUSD/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SHIB/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: UNI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MATIC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: BCH/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DAI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ATOM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: INJ/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ETH/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: LINK/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BLUR/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ICP/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: NEAR/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BAL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: GALA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: EURT/USDT | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: EGLD/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: OGN/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: 1INCH/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: STG/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: TRX/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: FORTH/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: MANA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SOL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: JUP/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: ZEC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: XMR/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DOT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: TURBO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: GRT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: FTM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: AVAX/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: FLOKI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: GNO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: SPELL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: XTZ/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BTC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: SUSHI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: LTC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: DASH/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ENA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: QTUM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BTT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: AAVE/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: MEME/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: DYM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: CRV/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: PEPE/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: WOO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: USDC/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: PORTAL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ADA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: APE/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: USDT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: TIA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: MLN/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: BOBA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: ARB/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: XRP/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0 
INFO:Pair: EURT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: KAVA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: BAT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: LDO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: WIF/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SNX/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SAND/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: OMG/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FET/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: BONK/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XLM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ZRO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: BAT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: LDO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: WIF/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SNX/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SAND/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: OMG/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FET/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: BONK/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XLM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ZRO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: APT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: SAND/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: OMG/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: FET/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: BONK/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: XLM/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: ZRO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
0
INFO:Pair: ZRO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ZRO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: APT/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: FIL/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: EOS/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: AXS/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: DOGE/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: COMP/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: SUI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: FLR/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: ALGO/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: CHZ/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 
0
INFO:Pair: YFI/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: JASMY/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0
INFO:Pair: LUNA/USD | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:Pair: EURT/EUR | Exchanges: ('kraken', 'bitfinex') | Total Profit: 0.00 | Total Trades: 0INFO:
Overall Total Profit: -8132506.74
INFO:Overall Total Trades: 563 """