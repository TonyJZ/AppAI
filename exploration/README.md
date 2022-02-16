## AppAI is the Appropolis AI library

## libraries installation  python, numpy, scipy, pandas, keras, statsmodels
1. install python3

2. check the installed modules
` /pip list` 

2. install pandas, numpy, matplotlib, statsmodels
` /pip install [module_name]`
`  /pip install pandas`

3. install fbprophet
 
 # install anaconda
 https://www.anaconda.com/distribution/#download-section

` /pip install pystan`
test code
>>> import pystan
>>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
>>> model = pystan.StanModel(model_code=model_code)
>>> y = model.sampling().extract()['y']
>>> y.mean()  # with luck the result will be near 0

` git clone https://github.com/facebookincubator/prophet`
` cd D:\Anaconda3\prophet\python `
` pip install -e . `
` pip install plotly`