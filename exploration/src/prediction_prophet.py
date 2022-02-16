import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from fbprophet import Prophet
# import plotly.graph_objs as go

from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
import plotly.offline as py
# py.init_notebook_mode()
# import plotly.offline as py
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    input_folder = './data/meters/'
    output_folder = './data/meters/chart/'
    for root, dirs, files in os.walk(input_folder):
        # root: current path
        # dirs: sub-directories
        # files: file names
        for name in files:
            df = pd.read_csv(input_folder + "/" + name)
            # Change date format to prophet datetime
            df['ds'] = pd.to_datetime(df['create_date'], format='%Y-%m-%d %H:%M:%S')
            df['y'] = df['ap']

            checkPts = pd.DataFrame(columns=['ds', 'y'])
            checkPts['y'] = df.loc[(df['ds'] > '2019-04-20') & (df['ds'] < '2019-04-30'), 'y']
            checkPts['ds'] = df.loc[(df['ds'] > '2019-04-20') & (df['ds'] < '2019-04-30'), 'ds']
            df.loc[(df['ds'] > '2019-04-20') & (df['ds'] < '2019-04-30'), 'y'] = None

            # set carrying capacity
            df['cap'] = 40

            # growth ='logistic' or 'linear'   specify the growth model
            
            # changepoint_range = 0.9  By default changepoints are only inferred for the first 80% of the time series 
            # in order to have plenty of runway for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series.
            
            # changepoint_prior_scale = 0.05 
            # If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), 
            # you can adjust the strength of the sparse prior using the input argument changepoint_prior_scale. 
            # By default, this parameter is set to 0.05. Increasing it will make the trend more flexible: 

            # holidays: data frame with columns holiday (character) and ds (date type)and optionally columns lower_window and upper_window 
            # which specify a range of days around the date to be included as holidays. 
            # lower_window=-2 will include 2 days prior to the date as holidays. Also optionally can have a column prior_scale specifying the prior scale for each holiday.
            
            # add holydays manually
            chinese_holydays = pd.DataFrame(
                {'holiday': 'Tomb-sweeping Day',
                'ds': pd.to_datetime(['2019-04-05']),
                'lower_window': 0,
                'upper_window': 2,}
                )

            m = Prophet(changepoint_prior_scale=0.01, changepoint_range=0.95, holidays=chinese_holydays, holidays_prior_scale = 0.05)

            # add default contry holidays
            # A list of available countries, and the country name to use, is available on their page: https://github.com/dr-prodigy/python-holidays. 
            # In addition to those countries, Prophet includes holidays for these countries: Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN), 
            # Thailand (TH), Philippines (PH), Turkey (TU), Pakistan (PK), Bangladesh (BD), Egypt (EG), China (CN), and Russian (RU).
            # m.add_country_holidays(country_name='CN')

            

            # add other seasonalities (monthly, quarterly, hourly) using the add_seasonality method
            # m = Prophet(weekly_seasonality=False)
            # m.add_seasonality(name='monthly', period=30.5, fourier_order=5) 

            m.fit(df) # training 

            future = m.make_future_dataframe(periods=240, freq='H')
            future['cap'] = 40
            # prediction
            fcst = m.predict(future, None)

            # prediction = pd.DataFrame(columns=['ds', 'yhat'])
            # prediction['ds'] = fcst.loc[checkPts['ds'], 'ds']
            # prediction['yhat'] = fcst.loc[checkPts['ds'], 'yhat']

            # diff = checkPts['y'] - prediction['yhat']

            with PdfPages(output_folder + "/" + name + "prophet.pdf") as pdf:
                # fig = plt.figure(figsize =(40,40))

                fig1 = m.plot(fcst)
                a = add_changepoints_to_plot(fig1.gca(), m, fcst)

                # sns.lineplot(ax = fig1, y="y", x = "ds", data = checkPts)
                plt.plot(checkPts['ds'], checkPts['y'], 'r.')

                fig2 = m.plot_components(fcst)

                pdf.savefig(fig1)
                pdf.savefig(fig2)

                fig3 = plt.figure()
                x = fcst['ds']
                y1 = fcst['yhat']
                y2 = fcst['yhat_lower']
                y3 = fcst['yhat_upper']
                diff = df['y'] - fcst['yhat']
                
                plt.plot(x,y1)
                plt.plot(x,y2)
                plt.plot(x,y3)
                plt.plot(x, diff, 'r.')
                # plt.show()
                pdf.savefig(fig3)

                plt.clf()
                plt.close()