"""
Flask Application for Glass Manufacturing Supply Chain Optimization
Allows users to:
1. Load Excel data
2. Explore and edit data
3. Run baseline optimization
4. Run stochastic analysis (Monte Carlo)
5. Tune safety stock parameters
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import os
import pandas as pd
import numpy as np
from pulp import *
from scipy import stats
from scipy.stats import kstest, shapiro, norm, gamma, lognorm, weibull_min, expon
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_in_production_12345'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for data (in production, use database or proper session management)
# This will store the current working data
current_data = {
    'demande_data': None,
    'other_data': None,
    'filename': None,
    'optimization_results': None,
    'stochastic_results': None,
    'fitted_distributions': None
}

# Make current_data available to all templates (used by base.html navbar)
@app.context_processor
def inject_current_data():
    return {'current_data': current_data}

# ============================================================================
# HELPER FUNCTIONS - Data Processing
# ============================================================================

def load_excel_data(filepath):
    """Load data from Excel file"""
    try:
        demande_data = pd.read_excel(filepath, sheet_name="Demandes", index_col='Modèles')
        other_data = pd.read_excel(filepath, sheet_name="DonneesSup", index_col='Modèles')
        demande_data = demande_data.transpose()
        return demande_data, other_data, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# HELPER FUNCTIONS - Baseline Optimization
# ============================================================================

def run_baseline_optimization(demande_data, other_data, safety_stock=None):
    """Run the baseline linear programming optimization"""
    try:
        weeks = list(range(1, 13))
        verres = list(other_data.index)
        
        # Initialize model
        model = LpProblem("Minimize_production_and_inventory_cost", LpMinimize)
        
        # Decision variables
        prod_i_j = LpVariable.dicts(
            name='Production_',
            indices=[(i, j) for i in verres for j in weeks],
            lowBound=0,
            cat=LpInteger
        )
        stock_i_j = LpVariable.dicts(
            name='Stockage_',
            indices=[(i, j) for i in verres for j in weeks],
            lowBound=0,
            cat=LpInteger
        )
        
        # Objective function
        model += (
            lpSum([other_data.loc[i, 'CoutProd'] * prod_i_j[i, j] for j in weeks for i in verres]) +
            lpSum([other_data.loc[i, 'CoutStock'] * stock_i_j[i, j] for j in weeks for i in verres])
        )
        
        # Constraints
        # Production capacity - TH
        for j in weeks:
            model += lpSum([other_data.loc[i, 'TH'] * prod_i_j[i, j] for i in verres]) <= 469
        
        # Production capacity - TM
        for j in weeks:
            model += lpSum([other_data.loc[i, 'TM'] * prod_i_j[i, j] for i in verres]) <= 914
        
        # Storage capacity - TS
        for j in weeks:
            model += lpSum([other_data.loc[i, 'TS'] * stock_i_j[i, j] for i in verres]) <= 1000
        
        # Inventory flow constraints
        for i in verres:
            # First week
            if safety_stock:
                # Add safety stock requirement
                model += stock_i_j[i, 1] == other_data.loc[i, 'StockInitial'] + prod_i_j[i, 1] - demande_data.loc[i, 1] + safety_stock.get(i, 0)
            else:
                model += stock_i_j[i, 1] == other_data.loc[i, 'StockInitial'] + prod_i_j[i, 1] - demande_data.loc[i, 1]
            
            # Subsequent weeks
            for j in weeks[1:]:
                if safety_stock:
                    model += stock_i_j[i, j] == stock_i_j[i, j-1] + prod_i_j[i, j] - demande_data.loc[i, j] + safety_stock.get(i, 0)
                else:
                    model += stock_i_j[i, j] == stock_i_j[i, j-1] + prod_i_j[i, j] - demande_data.loc[i, j]
            
            # Final stock constraint (matches notebook: pinned at week 3)
            model += stock_i_j[i, 3] == other_data.loc[i, 'StockFinal']
        
        # Solve
        model.solve(PULP_CBC_CMD(msg=0))
        
        # Extract results
        total_cost = value(model.objective)
        status = LpStatus[model.status]
        
        production = pd.DataFrame(
            {j: [value(prod_i_j[i, j]) for i in verres] for j in weeks},
            index=verres
        )
        stocks = pd.DataFrame(
            {j: [value(stock_i_j[i, j]) for i in verres] for j in weeks},
            index=verres
        )
        
        # Get constraint information
        constraints_info = []
        for nom, c in model.constraints.items():
            constraints_info.append({
                'name': nom,
                'pi': c.pi if c.pi else 0,
                'slack': c.slack if c.slack else 0
            })
        
        return {
            'success': True,
            'total_cost': total_cost,
            'status': status,
            'production': production.to_dict('index'),
            'stocks': stocks.to_dict('index'),
            'production_totals': {j: sum(production.loc[i, j] for i in verres) for j in weeks},
            'stock_totals': {j: sum(stocks.loc[i, j] for i in verres) for j in weeks},
            'constraints': constraints_info,
            'weeks': weeks,
            'verres': verres
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# HELPER FUNCTIONS - Distribution Fitting
# ============================================================================

def fit_distributions(demande_data, verres):
    """Fit distributions to demand data for each glass model"""
    try:
        fitted_distributions = {}
        
        distributions_to_test = {
            'normal': lambda x: (norm.fit(x), norm),
            'lognormal': lambda x: (lognorm.fit(x), lognorm),
            'gamma': lambda x: (gamma.fit(x), gamma),
            'weibull': lambda x: (weibull_min.fit(x), weibull_min),
            'expon': lambda x: (expon.fit(x), expon),
        }
        
        for verre in verres:
            data = demande_data.loc[verre].values
            
            best_fit = None
            best_p_value = -1
            best_params = None
            best_dist = None
            
            for dist_name, dist_func in distributions_to_test.items():
                try:
                    params, dist = dist_func(data)
                    ks_stat, p_value = kstest(data, lambda x: dist.cdf(x, *params))
                    
                    if p_value > best_p_value:
                        best_p_value = p_value
                        best_fit = dist_name
                        best_params = params
                        best_dist = dist
                except:
                    continue
            
            fitted_distributions[verre] = {
                'distribution': best_fit,
                'params': best_params,
                'dist_object': best_dist,
                'p_value': best_p_value,
                'mean': float(data.mean()),
                'std': float(data.std())
            }
        
        return fitted_distributions
        
    except Exception as e:
        return None

# ============================================================================
# HELPER FUNCTIONS - Monte Carlo Simulation
# ============================================================================

def generate_demand_sample(fitted_distributions, verres, weeks, seed=None):
    """Generate random demand samples from fitted distributions"""
    if seed is not None:
        np.random.seed(seed)
    
    demand_sample = {}
    
    for verre in verres:
        fit_info = fitted_distributions[verre]
        dist = fit_info['dist_object']
        params = fit_info['params']
        
        for week in weeks:
            sample = dist.rvs(*params)
            demand_sample[(verre, week)] = max(0, int(np.round(sample)))
    
    return demand_sample

def solve_with_uncertain_demand(demand_dict, verres, weeks, other_data):
    """Solve optimization with uncertain demand"""
    try:
        model = LpProblem("Optimize_with_uncertain_demand", LpMinimize)
        
        prod = LpVariable.dicts(
            name='Production_',
            indices=[(i, j) for i in verres for j in weeks],
            lowBound=0,
            cat=LpInteger
        )
        stock = LpVariable.dicts(
            name='Stock_',
            indices=[(i, j) for i in verres for j in weeks],
            lowBound=0,
            cat=LpInteger
        )
        
        # Objective
        model += (
            lpSum([other_data.loc[i, 'CoutProd'] * prod[i, j] for j in weeks for i in verres]) +
            lpSum([other_data.loc[i, 'CoutStock'] * stock[i, j] for j in weeks for i in verres])
        )
        
        # Capacity constraints
        for j in weeks:
            model += lpSum([other_data.loc[i, 'TH'] * prod[i, j] for i in verres]) <= 469
            model += lpSum([other_data.loc[i, 'TM'] * prod[i, j] for i in verres]) <= 914
        
        for j in weeks:
            model += lpSum([other_data.loc[i, 'TS'] * stock[i, j] for i in verres]) <= 1000
        
        # Flow constraints
        for i in verres:
            model += (stock[i, 1] == other_data.loc[i, 'StockInitial'] + prod[i, 1] - demand_dict[(i, 1)])
            
            for j in weeks[1:]:
                model += (stock[i, j] == stock[i, j-1] + prod[i, j] - demand_dict[(i, j)])
            
            model += stock[i, 3] == other_data.loc[i, 'StockFinal']
        
        # Solve (looser gap for Monte Carlo speed)
        model.solve(PULP_CBC_CMD(msg=0, gapRel=0.05, timeLimit=15))
        
        cost = value(model.objective)
        status = LpStatus[model.status]
        
        production_result = {}
        stock_result = {}
        for i in verres:
            for j in weeks:
                production_result[(i, j)] = value(prod[i, j])
                stock_result[(i, j)] = value(stock[i, j])
        
        return cost, production_result, stock_result, status
        
    except Exception as e:
        return None, None, None, str(e)

def run_monte_carlo(fitted_distributions, verres, weeks, other_data, num_iterations=100, seed=123):
    """Run Monte Carlo simulation"""
    try:
        np.random.seed(seed)
        
        results = {
            'iterations': [],
            'costs': [],
            'statuses': [],
            'productions': [],
            'stocks': [],
            'demands': []
        }
        
        for iteration in range(1, num_iterations + 1):
            # Generate demand sample
            demand_sample = generate_demand_sample(fitted_distributions, verres, weeks)
            
            # Solve optimization
            cost, prod_result, stock_result, status = solve_with_uncertain_demand(
                demand_sample, verres, weeks, other_data
            )
            
            if cost is not None:
                results['iterations'].append(iteration)
                results['costs'].append(float(cost))
                results['statuses'].append(status)
                results['productions'].append(prod_result)
                results['stocks'].append(stock_result)
                results['demands'].append(demand_sample)
        
        # Calculate statistics
        costs_array = np.array(results['costs'])
        
        stats = {
            'min': float(np.min(costs_array)),
            'max': float(np.max(costs_array)),
            'mean': float(np.mean(costs_array)),
            'median': float(np.median(costs_array)),
            'std': float(np.std(costs_array)),
            'cv_percent': float((np.std(costs_array) / np.mean(costs_array)) * 100),
            'p5': float(np.percentile(costs_array, 5)),
            'p25': float(np.percentile(costs_array, 25)),
            'p75': float(np.percentile(costs_array, 75)),
            'p95': float(np.percentile(costs_array, 95)),
            'optimal_count': sum(1 for s in results['statuses'] if s == 'Optimal'),
            'total_iterations': len(results['iterations'])
        }
        
        # Calculate decision statistics
        production_by_decision = {(i, j): [] for i in verres for j in weeks}
        stock_by_decision = {(i, j): [] for i in verres for j in weeks}
        
        for iteration in range(len(results['productions'])):
            prod_dict = results['productions'][iteration]
            stock_dict = results['stocks'][iteration]
            
            for key in production_by_decision.keys():
                production_by_decision[key].append(prod_dict[key])
                stock_by_decision[key].append(stock_dict[key])
        
        decision_stats = {}
        for key in production_by_decision.keys():
            prod_vals = np.array(production_by_decision[key])
            stock_vals = np.array(stock_by_decision[key])
            
            decision_stats[key] = {
                'production_mean': float(np.mean(prod_vals)),
                'production_std': float(np.std(prod_vals)),
                'production_p5': float(np.percentile(prod_vals, 5)),
                'production_p95': float(np.percentile(prod_vals, 95)),
                'stock_mean': float(np.mean(stock_vals)),
                'stock_std': float(np.std(stock_vals)),
                'stock_p5': float(np.percentile(stock_vals, 5)),
                'stock_p95': float(np.percentile(stock_vals, 95))
            }
        
        return {
            'success': True,
            'results': results,
            'statistics': stats,
            'decision_statistics': decision_stats
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# HELPER FUNCTIONS - Visualization
# ============================================================================

def create_matplotlib_figure():
    """Create a matplotlib figure"""
    fig = plt.figure(figsize=(10, 6))
    return fig

def save_figure_to_base64(fig):
    """Save matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def create_cost_distribution_plot(costs):
    """Create histogram of costs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(costs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(costs), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(costs):.2f}')
    ax.axvline(np.median(costs), color='green', linestyle='--', linewidth=2, label=f'Median: ${np.median(costs):.2f}')
    ax.set_xlabel('Total Cost ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Cost Distribution from Monte Carlo Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_boxplot_costs(costs):
    """Create boxplot of costs"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(costs, vert=True, patch_artist=True)
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Cost Distribution - Box Plot')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def create_cdf_plot(costs):
    """Create CDF plot of costs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_costs = np.sort(costs)
    cumulative_prob = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)
    ax.plot(sorted_costs, cumulative_prob, linewidth=2, color='darkblue')
    ax.fill_between(sorted_costs, cumulative_prob, alpha=0.3, color='lightblue')
    ax.set_xlabel('Total Cost ($)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function of Costs')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_production_bar_chart(production_df, verres, weeks):
    """Create bar chart of production by week"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, verre in enumerate(verres):
        if idx < len(axes):
            ax = axes[idx]
            weeks_to_plot = weeks[:6]
            values = [production_df.loc[verre, w] for w in weeks_to_plot]
            ax.bar(range(len(weeks_to_plot)), values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Week')
            ax.set_ylabel('Production')
            ax.set_title(f'Production Plan - {verre}')
            ax.set_xticks(range(len(weeks_to_plot)))
            ax.set_xticklabels(weeks_to_plot)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_stock_bar_chart(stocks_df, verres, weeks):
    """Create bar chart of inventory by week"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, verre in enumerate(verres):
        if idx < len(axes):
            ax = axes[idx]
            weeks_to_plot = weeks[:6]
            values = [stocks_df.loc[verre, w] for w in weeks_to_plot]
            ax.bar(range(len(weeks_to_plot)), values, color='lightcoral', alpha=0.7)
            ax.set_xlabel('Week')
            ax.set_ylabel('Inventory')
            ax.set_title(f'Inventory Plan - {verre}')
            ax.set_xticks(range(len(weeks_to_plot)))
            ax.set_xticklabels(weeks_to_plot)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page - File upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.xlsx'):
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Load data
        demande_data, other_data, error = load_excel_data(filepath)
        
        if error:
            flash(f'Error loading data: {error}', 'error')
            return redirect(url_for('index'))
        
        # Store in current_data
        current_data['demande_data'] = demande_data
        current_data['other_data'] = other_data
        current_data['filename'] = file.filename
        
        # Get glass models
        verres = list(other_data.index)
        
        flash('File uploaded and data loaded successfully!', 'success')
        return redirect(url_for('data_explorer'))
    else:
        flash('Please upload an Excel (.xlsx) file', 'error')
        return redirect(url_for('index'))

@app.route('/data_explorer')
def data_explorer():
    """Data exploration and editing page"""
    if current_data['demande_data'] is None or current_data['other_data'] is None:
        flash('Please upload data first', 'error')
        return redirect(url_for('index'))
    
    demande_data = current_data['demande_data']
    other_data = current_data['other_data']
    filename = current_data['filename']
    
    return render_template('data_explorer.html',
                         demande_data=demande_data.to_dict('index'),
                         other_data=other_data.to_dict('index'),
                         total_demand=int(demande_data.values.sum()),
                         avg_cout_prod=float(other_data['CoutProd'].mean()),
                         filename=filename,
                         verres=list(other_data.index),
                         weeks=list(demande_data.columns))

@app.route('/update_data', methods=['POST'])
def update_data():
    """Update data from editable tables"""
    if current_data['demande_data'] is None or current_data['other_data'] is None:
        return json.dumps({'success': False, 'error': 'No data loaded'})
    
    try:
        # Get updated values
        updated_demande = request.json.get('demande_data', {})
        updated_other = request.json.get('other_data', {})
        
        # Update current_data
        if updated_demande:
            # JSON object keys arrive as strings ("1".."12"); restore integer
            # week columns so optimization lookups (loc[model, week]) work
            fixed_demande = {
                verre: {int(week): value for week, value in weeks.items()}
                for verre, weeks in updated_demande.items()
            }
            current_data['demande_data'] = pd.DataFrame.from_dict(fixed_demande, orient='index')
        
        if updated_other:
            current_data['other_data'] = pd.DataFrame.from_dict(updated_other, orient='index')
        
        return json.dumps({'success': True, 'message': 'Data updated successfully'})
        
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})

@app.route('/baseline_optimization')
def baseline_optimization():
    """Run baseline optimization"""
    if current_data['demande_data'] is None or current_data['other_data'] is None:
        flash('Please upload data first', 'error')
        return redirect(url_for('index'))
    
    # Run optimization
    result = run_baseline_optimization(
        current_data['demande_data'],
        current_data['other_data']
    )
    
    if not result['success']:
        flash(f'Optimization error: {result["error"]}', 'error')
        return redirect(url_for('data_explorer'))
    
    # Store results
    current_data['optimization_results'] = result
    
    # Create visualizations
    production_df = pd.DataFrame.from_dict(result['production'], orient='index')
    stocks_df = pd.DataFrame.from_dict(result['stocks'], orient='index')
    
    prod_fig = create_production_bar_chart(production_df, result['verres'], result['weeks'])
    stock_fig = create_stock_bar_chart(stocks_df, result['verres'], result['weeks'])
    
    prod_plot = save_figure_to_base64(prod_fig)
    stock_plot = save_figure_to_base64(stock_fig)
    
    plt.close('all')
    
    return render_template('optimization.html',
                         result=result,
                         production_plot=prod_plot,
                         stock_plot=stock_plot)

@app.route('/stochastic_analysis')
def stochastic_analysis():
    """Run stochastic analysis"""
    if current_data['demande_data'] is None or current_data['other_data'] is None:
        flash('Please upload data first', 'error')
        return redirect(url_for('index'))
    
    verres = list(current_data['other_data'].index)
    weeks = list(range(1, 13))
    
    # Step 1: Fit distributions
    fitted_distributions = fit_distributions(current_data['demande_data'], verres)
    
    if fitted_distributions is None:
        flash('Error fitting distributions', 'error')
        return redirect(url_for('data_explorer'))
    
    current_data['fitted_distributions'] = fitted_distributions
    
    # Step 2: Run Monte Carlo simulation
    num_iterations = request.args.get('iterations', default=100, type=int)
    mc_result = run_monte_carlo(
        fitted_distributions,
        verres,
        weeks,
        current_data['other_data'],
        num_iterations=num_iterations,
        seed=123
    )
    
    if not mc_result['success']:
        flash(f'Monte Carlo error: {mc_result["error"]}', 'error')
        return redirect(url_for('data_explorer'))
    
    current_data['stochastic_results'] = mc_result
    
    # Create visualizations
    costs = mc_result['results']['costs']
    
    hist_fig = create_cost_distribution_plot(costs)
    box_fig = create_boxplot_costs(costs)
    cdf_fig = create_cdf_plot(costs)
    
    hist_plot = save_figure_to_base64(hist_fig)
    box_plot = save_figure_to_base64(box_fig)
    cdf_plot = save_figure_to_base64(cdf_fig)
    
    plt.close('all')
    
    return render_template('stochastic.html',
                         fitted_distributions=fitted_distributions,
                         mc_result=mc_result,
                         hist_plot=hist_plot,
                         box_plot=box_plot,
                         cdf_plot=cdf_plot,
                         verres=verres)

@app.route('/safety_stock', methods=['GET', 'POST'])
def safety_stock():
    """Safety stock parameter tuning"""
    if current_data['demande_data'] is None or current_data['other_data'] is None:
        flash('Please upload data first', 'error')
        return redirect(url_for('index'))
    
    verres = list(current_data['other_data'].index)
    
    if request.method == 'POST':
        # Get safety stock values from form
        safety_stock = {}
        for verre in verres:
            value = request.form.get(f'safety_stock_{verre}', '0')
            try:
                safety_stock[verre] = int(value)
            except:
                safety_stock[verre] = 0
        
        # Run optimization with safety stock
        result = run_baseline_optimization(
            current_data['demande_data'],
            current_data['other_data'],
            safety_stock=safety_stock
        )
        
        if result['success']:
            # Store result for comparison
            if 'safety_stock_results' not in current_data:
                current_data['safety_stock_results'] = []
            
            current_data['safety_stock_results'].append({
                'safety_stock': safety_stock,
                'result': result
            })
            
            # Create visualization
            production_df = pd.DataFrame.from_dict(result['production'], orient='index')
            prod_fig = create_production_bar_chart(production_df, result['verres'], result['weeks'])
            stock_fig = create_stock_bar_chart(pd.DataFrame.from_dict(result['stocks'], orient='index'), result['verres'], result['weeks'])
            
            prod_plot = save_figure_to_base64(prod_fig)
            stock_plot = save_figure_to_base64(stock_fig)
            
            plt.close('all')
            
            return render_template('safety_stock.html',
                                 verres=verres,
                                 result=result,
                                 safety_stock=safety_stock,
                                 production_plot=prod_plot,
                                 stock_plot=stock_plot,
                                 history=current_data.get('safety_stock_results', []))
    
    # GET request - show form
    return render_template('safety_stock.html',
                         verres=verres,
                         result=None,
                         safety_stock=None,
                         production_plot=None,
                         stock_plot=None,
                         history=current_data.get('safety_stock_results', []))

@app.route('/reset_data')
def reset_data():
    """Reset all data and start over"""
    current_data['demande_data'] = None
    current_data['other_data'] = None
    current_data['filename'] = None
    current_data['optimization_results'] = None
    current_data['stochastic_results'] = None
    current_data['fitted_distributions'] = None
    current_data['safety_stock_results'] = []
    
    flash('All data has been reset', 'info')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
