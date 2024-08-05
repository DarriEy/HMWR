import xarray as xr # type: ignore
import pandas as pd # type: ignore
from utils.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore

def evaluate_model(mizuroute_output_path, calib_period, eval_period, sim_reach_ID, obs_file_path):
    """
    Evaluate the model by comparing simulations to observations.

    Parameters:
    mizuroute_rank_specific_path (Path): Path to mizuRoute output
    calib_period (tuple): Start and end dates for calibration period
    eval_period (tuple): Start and end dates for evaluation period
    sim_reach_ID (str): ID of the simulated reach to evaluate
    obs_file_path (str): Path to the observation file

    Returns:
    tuple: Calibration and evaluation metrics
    """

    # Open the mizuRoute output file
    #sim_file_path = str(mizuroute_rank_specific_path) + '/*.nc'
    dfSim = xr.open_dataset(mizuroute_output_path)         
    segment_index = dfSim['reachID'].values == int(sim_reach_ID)
    dfSim = dfSim.sel(seg=segment_index) 
    dfSim = dfSim['IRFroutedRunoff'].to_dataframe().reset_index()
    dfSim.set_index('time', inplace=True)

    dfObs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
    dfObs = dfObs['discharge_cms'].resample('h').mean()

    def calculate_metrics(obs, sim):
        return {
            'RMSE': get_RMSE(obs, sim, transfo=1),
            'KGE': get_KGE(obs, sim, transfo=1),
            'KGEp': get_KGEp(obs, sim, transfo=1),
            'NSE': get_NSE(obs, sim, transfo=1),
            'MAE': get_MAE(obs, sim, transfo=1)
        }

    calib_start, calib_end = calib_period
    calib_obs = dfObs.loc[calib_start:calib_end]
    calib_sim = dfSim.loc[calib_start:calib_end]
    calib_metrics = calculate_metrics(calib_obs.values, calib_sim['IRFroutedRunoff'].values)

    eval_start, eval_end = eval_period
    eval_obs = dfObs.loc[eval_start:eval_end]
    eval_sim = dfSim.loc[eval_start:eval_end]
    eval_metrics = calculate_metrics(eval_obs.values, eval_sim['IRFroutedRunoff'].values)

    return calib_metrics, eval_metrics
    