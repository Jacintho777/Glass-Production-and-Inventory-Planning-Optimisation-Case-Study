# Supply Chain Optimization - Flask Web Application

A comprehensive web application for optimizing glass manufacturing production planning under demand uncertainty. This application extends the original Jupyter notebook case study into an interactive web interface.

## Features

### 1. **Data Management**
- Upload Excel files containing demand and parameter data
- Interactive data exploration with tabular views
- Edit data values directly in the browser
- Save changes and re-run analysis

### 2. **Baseline Optimization**
- Linear programming model using PuLP
- Minimize total production and inventory costs
- Visualize production and inventory plans
- Constraint analysis with shadow prices

### 3. **Stochastic Analysis**
- **Distribution Fitting**: Automatically fit distributions (Normal, Lognormal, Gamma, Weibull, Exponential) to historical demand data
- **Monte Carlo Simulation**: Run 100+ iterations with randomly sampled demand
- **Statistical Analysis**: Compute percentiles, mean, median, standard deviation, coefficient of variation
- **Visualization**: Histograms, box plots, CDF curves
- **Decision Analysis**: Analyze how production and inventory decisions vary across scenarios

### 4. **Safety Stock Tuning**
- Configure safety stock levels for each glass model
- Run optimization with safety stock constraints
- Compare different configurations
- History tracking for trial-and-error tuning

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd flask_app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Access the application:**
Open your browser and go to: `http://localhost:5000`

## File Structure

```
flask_app/
├── app.py                 # Main Flask application with all routes and logic
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   ├── base.html          # Base template with navigation and styling
│   ├── index.html         # File upload page
│   ├── data_explorer.html # Data exploration and editing
│   ├── optimization.html  # Baseline optimization results
│   ├── stochastic.html    # Monte Carlo simulation results
│   └── safety_stock.html  # Safety stock parameter tuning
├── static/
│   ├── css/               # Custom CSS (optional)
│   └── js/                # Custom JavaScript (optional)
└── uploads/               # Uploaded Excel files
```

## Data Format

The application expects an Excel file with two sheets:

### Sheet 1: `Demandes`
- Index column: `Modèles` (glass model names)
- Columns: Week 1, Week 2, ..., Week 12 (demand quantities for each week)
- Example structure: 6 glass models × 12 weeks

### Sheet 2: `DonneesSup`
- Index column: `Modèles` (glass model names - must match Demandes sheet)
- Columns:
  - `CoutProd`: Production cost per unit
  - `CoutStock`: Storage cost per unit per week
  - `TH`: Labor hours required per unit
  - `TM`: Machine time required per unit
  - `TS`: Storage space required per unit
  - `StockInitial`: Initial inventory
  - `StockFinal`: Required final inventory

## Usage Workflow

1. **Upload Data**
   - Navigate to the home page
   - Upload your Excel file
   - The application will automatically load both sheets

2. **Explore & Edit Data**
   - View demand data and parameters in tabular format
   - Edit any cell by clicking on it
   - Save changes to update the analysis

3. **Run Baseline Optimization**
   - Click "Run Optimization" to execute the deterministic model
   - View production and inventory plans
   - Analyze constraint slack and shadow prices

4. **Run Stochastic Analysis**
   - Click "Stochastic Analysis" to run Monte Carlo simulation
   - View distribution fitting results
   - Analyze cost distribution and decision variability
   - Identify risk levels and uncertainty impact

5. **Tune Safety Stock**
   - Navigate to "Safety Stock Tuning"
   - Set safety stock levels for each glass model
   - Run optimization with safety stock constraints
   - Compare different configurations
   - Use history to track trial-and-error process

## Technical Details

### Optimization Model

The application implements a multi-period production planning model:

**Objective:** Minimize total production + inventory costs

**Decision Variables:**
- `prod_i_j`: Production quantity for glass model i in week j
- `stock_i_j`: Inventory level for glass model i in week j

**Constraints:**
- Production capacity (labor hours): TH ≤ 469 per week
- Production capacity (machine time): TM ≤ 914 per week
- Storage capacity: TS ≤ 1000 per week
- Inventory flow: stock[i,j] = stock[i,j-1] + prod[i,j] - demand[i,j] + safety_stock[i]
- Final inventory: stock[i,12] = StockFinal[i]

### Stochastic Model

1. **Distribution Fitting**: Uses Kolmogorov-Smirnov test to select best-fit distribution
2. **Monte Carlo**: Generates random demand samples and solves optimization for each
3. **Statistics**: Computes comprehensive statistics across all iterations

### Dependencies

- **Flask**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **PuLP**: Linear programming
- **scipy**: Statistical functions and distribution fitting
- **matplotlib**: Visualization
- **openpyxl**: Excel file reading

## Configuration

### Environment Variables

Create a `.env` file for production:
```bash
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your_secret_key_here
UPLOAD_FOLDER=./uploads
```

### Customization

You can customize:
- Number of Monte Carlo iterations (edit `run_monte_carlo` function)
- Random seed for reproducibility
- Capacity constraints (TH, TM, TS limits)
- Visualization styles

## Troubleshooting

### Common Issues

1. **File upload fails**: Ensure the Excel file has the correct sheets (Demandes, DonneesSup)
2. **Optimization timeout**: Reduce problem size or simplify constraints
3. **Memory issues**: For large Monte Carlo runs, reduce iteration count
4. **Missing dependencies**: Run `pip install -r requirements.txt`

### Tips

- Start with the original `case_study3.xlsx` to test the application
- Use small numbers for initial testing
- The Monte Carlo simulation with 100 iterations may take 1-2 minutes
- For faster results, reduce the number of iterations

## Security Notes

- The application uses in-memory storage for data (not suitable for production)
- For production use, implement:
  - Database storage for uploaded files and results
  - User authentication
  - Proper session management
  - File size limits and validation

## License

This project is provided as-is for educational and research purposes.
