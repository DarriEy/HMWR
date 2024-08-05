# results_utils.py
import os
import csv
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd # type: ignore
import numpy as np # type: ignore
from utils.config import Config # type: ignore
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
import seaborn as sns # type: ignore
import plotly.graph_objects as go # type: ignore

class Results:
    def __init__(self, config: 'Config', logger: 'logging.Logger'):
        self.config = config
        self.logger = logger
        self.iteration_count = 0
        self.iteration_results_file: Optional[str] = None
        self.results_df = pd.DataFrame()
        self.output_folder = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "plots" / "calibration diagnostics"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def create_iteration_results_file(self):
        """Create a new iteration results file with appropriate headers."""
        iteration_results_dir = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "optimisation"
        iteration_results_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_results_file = str(iteration_results_dir / f"{self.config.experiment_id}_parallel_iteration_results.csv")
        
        Path(self.iteration_results_file).touch()
        self.logger.info(f"Created iteration results file: {self.iteration_results_file}")
        return self.iteration_results_file

    def plot_objective_evolution(self, in_progress: bool = False) -> None:
        """Plot the evolution of objective metrics."""
        calib_metrics = ['KGE', 'KGEnp', 'KGEp', 'MAE', 'NSE', 'RMSE']
        eval_metrics = ['Eval_KGE', 'Eval_KGEnp', 'Eval_KGEp', 'Eval_MAE', 'Eval_NSE', 'Eval_RMSE']
        all_metrics = calib_metrics + eval_metrics
        
        available_metrics = [m for m in all_metrics if m in self.results_df.columns]
        
        if not available_metrics:
            self.logger.warning("No metrics available for plotting. Skipping objective evolution plot.")
            return

        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 4*len(available_metrics)), sharex=True)
        if len(available_metrics) == 1:
            axes = [axes]  # Make axes always iterable
        
        for ax, metric in zip(axes, available_metrics):
            ax.plot(self.results_df['Iteration'], self.results_df[metric], 'b-', alpha=0.5, label='All iterations')
            
            if metric.endswith(('RMSE', 'MAE')):
                # For these metrics, lower is better
                ax.plot(self.results_df['Iteration'], self.results_df[metric].cummin(), 'r-', label='Best so far')
                best_value = self.results_df[metric].min()
            else:
                # For other metrics (like KGE, NSE), higher is better
                ax.plot(self.results_df['Iteration'], self.results_df[metric].cummax(), 'r-', label='Best so far')
                best_value = self.results_df[metric].max()
            
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()
            
            ax.text(0.02, 0.95, f'Best: {best_value:.4f}', transform=ax.transAxes, 
                    verticalalignment='top', fontweight='bold')
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Evolution of Objective Metrics Over Iterations')
        plt.tight_layout()
        
        suffix = '_in_progress' if in_progress else '_final'
        plt.savefig(self.output_folder / f'objective_evolution{suffix}.png')
        plt.close()

    def update_results(self, params: List[float], result: Dict[str, Any]) -> None:
        """Update results with new data and save to file."""
        new_row = pd.DataFrame({
            **dict(zip(self.config.all_params, params)),
            **result.get('calib_metrics', {}),
            **result.get('eval_metrics', {}),
            'Iteration': self.iteration_count
        }, index=[0])
        
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        # Save updated results to file
        self.results_df.to_csv(self.iteration_results_file, index=False)
        
        if self.iteration_count % self.config.diagnostic_frequency == 0:
            self.generate_in_progress_diagnostics()

    def log_iteration_results(self, params: List[float], result: Dict[str, Any]) -> None:
        """Log the results of each iteration."""
        self.iteration_count += 1
        calib_metrics = result.get('calib_metrics', {}) or {}
        eval_metrics = result.get('eval_metrics', {}) or {}
        
        new_row = pd.DataFrame({
            **dict(zip(self.config.all_params, params)),
            **calib_metrics,
            **{f'Eval_{k}': v for k, v in eval_metrics.items()},
            'Iteration': self.iteration_count
        }, index=[0])
        
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
        
        self.write_iteration_results()
        
        #if self.iteration_count % self.config.diagnostic_frequency == 0:
            #self.generate_in_progress_diagnostics()

    def write_iteration_results(self) -> None:
        """Write iteration results to the CSV file."""
        try:
            self.results_df.to_csv(self.iteration_results_file, index=False)
            self.logger.info(f"Results written successfully for iteration {self.iteration_count}")
        except Exception as e:
            self.logger.error(f"Error writing results for iteration {self.iteration_count}: {str(e)}")

    def save_pareto_optimal_solutions(self, best_params: List[List[float]], best_values: List[List[float]]) -> None:
        """
        Save the Pareto optimal solutions to a file.

        Args:
            best_params (List[List[float]]): List of Pareto optimal parameter sets.
            best_values (List[List[float]]): List of objective values for Pareto optimal solutions.
        """
        with open(self.config.optimization_results_file, 'w') as f:
            f.write(f"{self.config.algorithm} Optimization Results:\n")
            for i, (params, values) in enumerate(zip(best_params, best_values)):
                f.write(f"\nSolution {i+1}:\n")
                for param, value in zip(self.config.all_params, params):
                    f.write(f"{param}: {value:.6e}\n")
                f.write(f"Objective Values: {values}\n")
                
                # Note: We can't evaluate final metrics here as it requires running the model
                # This part might need to be handled differently or moved elsewhere

    def write_optimization_results(self, best_params: List[float], final_result: Dict[str, Any]) -> None:
        """
        Write optimization results to a file.

        Args:
            best_params (List[float]): List of best parameter values found during optimization.
            final_result (Dict[str, Any]): Dictionary containing final calibration and evaluation metrics.
        """
        results_file = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "optimisation" / "calibration_results.txt"
        with open(results_file, 'w') as f:
            f.write("Optimization Results:\n")
            f.write(f"Algorithm: {self.config.algorithm}\n")
            f.write("Best parameters:\n")
            for param, value in zip(self.config.all_params, best_params):
                f.write(f"{param}: {value:.6e}\n")
            f.write(f"Final performance metrics:\n")
            f.write(f"Calibration: {final_result['calib_metrics']}\n")
            f.write(f"Evaluation: {final_result['eval_metrics']}\n")

    def generate_in_progress_diagnostics(self) -> None:
            """Generate and save diagnostics during the optimization process."""
            self.plot_objective_evolution(in_progress=True)
            self.plot_parameter_convergence(in_progress=True)

    def generate_final_diagnostics(self) -> None:
        """Generate and save final comprehensive diagnostics."""
        self.plot_objective_evolution()
        self.plot_parameter_convergence()
        self.plot_parallel_coordinates()
        self.plot_pairwise_scatter()
        self.plot_best_vs_worst()
        self.plot_parameter_correlation()
        self.plot_objective_surface()

    def save_final_results(self) -> None:
        """Save the final results dataframe to a CSV file."""
        self.results_df.to_csv(self.output_folder / 'final_results.csv', index=False)

    def plot_comparison(self, ax: plt.Axes, sim_data: xr.DataArray, obs_data: pd.Series, 
                        period: Tuple[str, str], title: str) -> None:
        """
        Plot the comparison between simulated and observed streamflow for a given period.

        Args:
            ax (plt.Axes): The axes to plot on.
            sim_data (xr.DataArray): The simulated streamflow data.
            obs_data (pd.Series): The observed streamflow data.
            period (Tuple[str, str]): The start and end dates of the period to plot.
            title (str): The title for the subplot.
        """
        start_date, end_date = period
        
        # Slice the data for the specified period
        sim_period = sim_data.sel(time=slice(start_date, end_date))
        obs_period = obs_data.loc[start_date:end_date]

        # Plot the data
        ax.plot(sim_period.time, sim_period.values, label='Simulated', color='blue')
        ax.plot(obs_period.index, obs_period.values, label='Observed', color='red')

        ax.set_ylabel('Streamflow (cms)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    def plot_best_simulation_vs_observation(self) -> None:
        """
        Plot the time series of the best simulation against observations for both
        calibration and evaluation periods.
        """
        # Identify the best parameter set
        best_metric = f"Calib_{self.config.optimization_metric}"
        if self.config.optimization_metric in ['RMSE', 'MAE']:
            best_params = self.results_df.loc[self.results_df[best_metric].idxmin()]
        else:
            best_params = self.results_df.loc[self.results_df[best_metric].idxmax()]

        # Load the simulation data for the best parameter set
        sim_data = self.load_simulation_data(best_params)

        # Load the observation data
        obs_data = self.load_observation_data()

        # Plot the comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Calibration period
        self.plot_comparison(ax1, sim_data, obs_data, self.config.calib_period, "Calibration Period")

        # Evaluation period
        self.plot_comparison(ax2, sim_data, obs_data, self.config.eval_period, "Evaluation Period")

        plt.xlabel('Date')
        fig.suptitle('Best Simulation vs Observation', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'best_simulation_vs_observation.png')
        plt.close()

    def load_simulation_data(self, best_params: pd.Series) -> xr.DataArray:
        """
        Load the simulation data for the best parameter set.

        Args:
            best_params (pd.Series): Series containing the best parameter values.

        Returns:
            xr.DataArray: DataArray containing the simulated streamflow.
        """
        # Construct the path to the simulation output file
        sim_file = Path(self.config.root_path) / f"domain_{self.config.domain_name}" / "simulations" / \
                   f"{self.config.experiment_id}_best" / "mizuRoute" / f"{self.config.experiment_id}_best.h.*.nc"
        
        # Load the netCDF file
        ds = xr.open_mfdataset(sim_file, combine='by_coords')
        
        # Extract the simulated streamflow for the specified reach
        sim_flow = ds.sel(seg=self.config.sim_reach_ID)['IRFroutedRunoff']
        
        return sim_flow

    def load_observation_data(self) -> pd.Series:
        """
        Load the observation data.

        Returns:
            pd.Series: Series containing the observed streamflow.
        """
        # Load the CSV file
        obs_df = pd.read_csv(self.config.obs_file_path, parse_dates=['datetime'], index_col='datetime')
        
        # Extract the streamflow column
        obs_flow = obs_df['discharge_cms']
        
        return obs_flow    
    
    def plot_parameter_correlation(self) -> None:
        corr_columns = self.config.all_params + self.calib_metrics + self.eval_metrics
        corr_matrix = self.results_df[corr_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Parameter and Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_folder / 'parameter_correlation.png')
        plt.close()

    def plot_objective_surface(self, n_levels: int = 20) -> None:
        metric = self.config.optimization_metric
        if not metric.startswith(('Calib_', 'Eval_')):
            metric = f'Calib_{metric}'
        
        parameter_columns = self.config.all_params
        sensitivities = [(param, abs(self.results_df[param].corr(self.results_df[metric]))) for param in parameter_columns]
        top_two_params = sorted(sensitivities, key=lambda x: x[1], reverse=True)[:2]
        
        param1, param2 = top_two_params[0][0], top_two_params[1][0]
        
        plt.figure(figsize=(10, 8))
        contour = plt.tricontourf(self.results_df[param1], self.results_df[param2], self.results_df[metric], levels=n_levels, cmap='viridis')
        plt.colorbar(contour, label=metric)
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Objective Function Surface for Top 2 Sensitive Parameters')
        plt.tight_layout()
        plt.savefig(self.output_folder / 'objective_surface.png')
        plt.close()
    
    def plot_pairwise_scatter(self) -> None:
        plot_columns = self.config.all_params + [self.config.optimization_metric]
        if not self.config.optimization_metric.startswith(('Calib_', 'Eval_')):
            plot_columns[-1] = f'Calib_{self.config.optimization_metric}'
        
        g = sns.PairGrid(self.results_df[plot_columns])
        g.map_upper(sns.scatterplot, alpha=0.5)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot, kde=True)

        plt.suptitle('Pairwise Relationships Between Parameters and Objective Function', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'pairwise_scatter.png', bbox_inches='tight')
        plt.close()

    def plot_best_vs_worst(self) -> None:
        """Compare best and worst solutions based on the optimization metric."""
        metric = self.config.optimization_metric
        best_solution = self.results_df.loc[self.results_df[metric].idxmin() if metric in ['RMSE', 'MAE'] else self.results_df[metric].idxmax()]
        worst_solution = self.results_df.loc[self.results_df[metric].idxmax() if metric in ['RMSE', 'MAE'] else self.results_df[metric].idxmin()]
        
        n_params = len(self.config.all_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten()

        for i, param in enumerate(self.config.all_params):
            ax = axes[i]
            best_val = best_solution[param]
            worst_val = worst_solution[param]
            
            ax.bar([0, 1], [best_val, worst_val], color=['g', 'r'], alpha=0.7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Best', 'Worst'])
            ax.set_title(param)
            ax.tick_params(axis='x', rotation=45)
            
            if param in self.config.all_bounds:
                lower, upper = self.config.all_bounds[param]
                ax.axhline(y=lower, color='k', linestyle='--', alpha=0.5)
                ax.axhline(y=upper, color='k', linestyle='--', alpha=0.5)
            
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.suptitle(f'Comparison of Best and Worst Solutions (based on {metric})', y=1.02)
        plt.savefig(self.output_folder / 'best_vs_worst.png', bbox_inches='tight')
        plt.close()

    def plot_parameter_convergence(self, in_progress: bool = False) -> None:
        parameter_columns = self.config.all_params
        fig, axes = plt.subplots(len(parameter_columns), 1, figsize=(12, 4*len(parameter_columns)), sharex=True)
        
        for ax, param in zip(axes, parameter_columns):
            ax.plot(self.results_df['Iteration'], self.results_df[param], 'b.')
            ax.set_ylabel(param)
            ax.grid(True)
            
            if param in self.config.all_bounds:
                lower, upper = self.config.all_bounds[param]
                ax.axhline(y=lower, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=upper, color='r', linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle('Parameter Values Over Iterations')
        plt.tight_layout()
        
        suffix = '_in_progress' if in_progress else '_final'
        plt.savefig(self.output_folder / f'parameter_convergence{suffix}.png')
        plt.close()

    def plot_parallel_coordinates(self) -> None:
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = self.results_df[self.config.optimization_metric],
                            colorscale = 'Viridis',
                            showscale = True),
                dimensions = [dict(label = col, values = self.results_df[col]) for col in self.config.all_params + [self.config.optimization_metric]]
            )
        )
        
        fig.update_layout(
            title='Parallel Coordinates Plot of Parameters and Objective Function',
            width=1200,
            height=600
        )
        
        fig.write_html(self.output_folder / 'parallel_coordinates.html')

    def plot_parameter_correlation(self) -> None:
        """Generate a correlation heatmap for parameters and metrics."""
        corr_columns = self.config.all_params + ['RMSE', 'KGE', 'KGEp', 'KGEnp', 'NSE', 'MAE']
        corr_matrix = self.results_df[corr_columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Parameter and Metric Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_folder / 'parameter_correlation.png')
        plt.close()

    def plot_objective_surface(self, n_levels: int = 20) -> None:
        """
        Plot the objective function surface for the two most sensitive parameters.

        Args:
            n_levels (int): Number of contour levels to plot.
        """
        metric = self.config.optimization_metric
        parameter_columns = self.config.all_params
        sensitivities = [(param, abs(self.results_df[param].corr(self.results_df[metric]))) for param in parameter_columns]
        top_two_params = sorted(sensitivities, key=lambda x: x[1], reverse=True)[:2]
        
        param1, param2 = top_two_params[0][0], top_two_params[1][0]
        
        plt.figure(figsize=(10, 8))
        contour = plt.tricontourf(self.results_df[param1], self.results_df[param2], self.results_df[metric], levels=n_levels, cmap='viridis')
        plt.colorbar(contour, label=metric)
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f'Objective Function Surface for Top 2 Sensitive Parameters')
        plt.tight_layout()
        plt.savefig(self.output_folder / 'objective_surface.png')
        plt.close()    