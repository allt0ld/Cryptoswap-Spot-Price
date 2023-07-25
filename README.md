# Cryptoswap-Spot-Price
We differentiate Curve v2's Cryptoswap invariant to derive a spot price formula and compare it to the "get_p" function currently implemented in [Tricrypto-NG pools](https://curve.fi/#/ethereum/pools/factory-tricrypto-0/deposit). We output Cryptoswap parameters and test results in a csv file, and we plot two histograms of the differences between an effective swap price and each spot price formula and one histogram of the difference between each formula's difference.

## Usage
```
Cryptoswap-Spot.py [-h] [-f | -n | -s COLUMNS] [-d DIRECTORY] [-p]
                          [-q | -v]

  -h, --help            show this help message and exit
  -f, --full            all columns to csv output
  -n, --nothing         no csv output
  -s COLUMNS, --select COLUMNS
                        ADVANCED: select columns by label for csv output.
                        Format: "'label1', 'label2', ...," (" " outside and '
                        ' inside; writing "[...]" or "(...)" is optional)
  -d DIRECTORY, --dest DIRECTORY
                        select csv output directory
  -p, --plot            output histograms
  -q, --quiet           print nothing
  -v, --verbose         increase print verbosity
```
