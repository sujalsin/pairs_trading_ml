# Machine Learning-Enhanced Pairs Trading Strategy

A sophisticated implementation of a pairs trading strategy that leverages machine learning and statistical arbitrage techniques to identify and exploit relative mispricings between correlated stocks.

## Strategy Overview

```
                                   ML-Enhanced Pairs Trading
                                           Flow
                                            
    ┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
    │  Data Pipeline  │     │ Pairs Select  │     │   ML Model     │
    │  & Processing   │ ──> │& Cointegration│ ──> │   Training     │
    └─────────────────┘     └──────────────┘     └────────────────┘
            │                      │                     │
            │                      │                     │
            v                      v                     v
    ┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
    │    Real-time    │     │   Signal     │     │   Position     │
    │  Data Updates   │ ──> │  Generation  │ ──> │   Management   │
    └─────────────────┘     └──────────────┘     └────────────────┘
```

## Example Scenarios

### Scenario 1: Mean Reversion Trade
```
Price Spread
    │      ╭─── Entry Short (Z-score > 0.5)
    │    ╭─╯
    │   ╭╯
    │  ╭╯     Mean
────┴──╯─────────────────────────────
    │╭╯
    │╯
    ╯  ╰─── Entry Long (Z-score < -0.5)
    Time
```

### Scenario 2: ML-Enhanced Signal Generation
```
                   High Confidence Zone
                   ┌──────────────┐
Prediction  1.0 ─  │ LONG        │
Confidence        │              │
           0.0 ─  │──────────────│  ─ Neutral Zone
                  │              │
          -1.0 ─  │ SHORT       │
                   └──────────────┘
                   Low Confidence Zone
```

### Scenario 3: Risk Management
```
Position Size
    │                 Max Position
    │    ╭─────────────────────────
    │   ╱
    │  ╱
    │ ╱
    │╱
    ╱│
   ╱ │
──╯  │
    Signal Strength
```

## Project Structure

```
pairs_trading_ml/
├── data/                      # Data storage directory
├── src/                      # Source code
│   ├── data/                # Data collection and preprocessing
│   ├── models/              # ML models and predictive analytics
│   ├── strategy/            # Trading strategy implementation
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for analysis
└── config/                  # Configuration files
```

## Features

- Data collection and preprocessing pipeline
- Statistical analysis for pairs selection
- Machine learning models for spread prediction
- Real-time trading strategy implementation
- Comprehensive backtesting framework
- Risk management and position sizing
- Performance analytics and reporting

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pairs_trading_ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Collection:
```bash
python src/data/collect_data.py --symbols-file config/symbols.txt --start-date 2010-01-01
```

2. Pairs Selection:
```bash
python src/strategy/pairs_selection.py --data-dir data/processed --output-dir data/pairs
```

3. Model Training:
```bash
python src/models/train.py --pairs-file data/pairs/selected_pairs.csv --model-type "ensemble"
```

4. Backtesting:
```bash
python src/strategy/backtest.py --model-path models/trained_model.pkl --pairs-file data/pairs/selected_pairs.csv
```

## Configuration

The strategy parameters can be configured in `config/strategy_config.yaml`:

- Formation period length
- Trading period length
- Z-score thresholds
- Stop-loss limits
- Position sizing rules
- Risk management parameters

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
