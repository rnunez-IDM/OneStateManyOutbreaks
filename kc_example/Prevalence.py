""" Prevalence.py

This scipt follows the order in the technical report: Reff is estimated, the IFR is estimated (based
on the age pyramid, with or without transient correction), initial conditions are fit to
mortality, and then finally reporting rates are estimated (in conjuction with plotting because 
reporting rate estimates are really only used for fit to cases plotting). 

Hyper parameters are set in json style at the top, and they're used in an otherwise agnostic
pipeline below.

NB: Because of the way mortality is fit, this pipeline will FAIL if there's unknown importations
at the same time periods were Reff is being inferred. KNOWN importations during those periods can
be incorporated. """
import sys
sys.path.append("..\\")

## Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## For time-series modeling and Reff estimation
from rainier.seir import LogNormalSEIR, SampleMortality,\
						 DeathRSS, MeanHosp
from rainier.rainier import continuous_time_posterior,\
							SplineTestingEpiCurve,\
							RandomWalkEpiCurve,\
							GaussianProcessIFR,\
							GaussianProcessIHR
from rainier.stats import WeightedLeastSquares

## For feedback and progress throughout
from tqdm import tqdm

## For histograms
from scipy.stats import gaussian_kde

## For optimization
from scipy.optimize import minimize
from scipy.special import gammaln

## Hyper parameters and additional options
np.random.seed(6)
hp = {"counties":["king"],
	  "tr_start":"2020-03-01",
	  "tr_end":"2020-09-11",
	  "unabated_date":"2020-03-01",
	  "gp_ifr":True,
	  "pulses":["2020-01-15","2020-02-01"],
	  "reporting_regimes":["2020-06-07","2020-06-28"],
	  "reff_output":None,#"kc_r0_9_20.pkl",
	  "reff_old":None,#"kc_r0_9_13.pkl",
	  "prev_output":None,#"king_prev_9_6.pkl",
	  "plot_time_end":"2020-09-14",
	  "model_pickle":None,#"king_model.pkl",
	  }

## Get the probability of death, via
## https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30243-7/fulltext
## Table 1
ifr_table = pd.DataFrame([("0 to 9",0.0000161,0.00000185,0.000249),
						  ("10 to 19",0.0000695,0.0000149,0.000502),
						  ("20 to 29",0.000309,0.000138,0.000923),
						  ("30 to 39",0.000844,0.000408,0.00185),
						  ("40 to 49",0.00161,0.000764,0.00323),
						  ("50 to 59",0.00595,0.00344,0.0128),
						  ("60 to 69",0.0193,0.0111,0.0389),
						  ("70 to 79",0.0428,0.0245,0.0844),
						  ("over 80",0.0780,0.0380,0.133)],
						  columns=["age","mid","low","high"]).set_index("age")

## Get the probability infections are hospitalized
## via Table 3
ihr_table = pd.DataFrame([("0 to 9",0,0,0),
						  ("10 to 19",0.000408,0.000243,0.000832),
						  ("20 to 29",0.0104,0.00622,0.0213),
						  ("30 to 39",0.0343,0.0204,0.0700),
						  ("40 to 49",0.0425,0.0253,0.0868),
						  ("50 to 59",0.0816,0.0486,0.167),
						  ("60 to 69",0.118,0.0701,0.24),
						  ("70 to 79",0.166,0.0987,0.338),
						  ("over 80",0.184,0.11,0.376)],
						  columns=["age","mid","low","high"]).set_index("age")

## Helper functions
def axes_setup(axes):
	axes.spines["left"].set_position(("axes",-0.025))
	axes.spines["top"].set_visible(False)
	axes.spines["right"].set_visible(False)
	return

def low_mid_high(samples):
	l0 = np.percentile(samples,1.,axis=0)
	h0 = np.percentile(samples,99.,axis=0)
	l1 = np.percentile(samples,2.5,axis=0)
	h1 = np.percentile(samples,97.5,axis=0)
	l2 = np.percentile(samples,25.,axis=0)
	h2 = np.percentile(samples,75.,axis=0)
	return l0, h0, l1, h1, l2, h2

def GetPopulationPyramid(group):

	## Get the CSV from Roy
	age_pop = pd.read_csv("..\\_example_datasets\\WA_county_age_pop.csv",
						  header=0,
						  usecols=["county","age_grp10","pop","pop_pct"])

	## Make some adjustments
	age_pop["county"] = age_pop["county"].str.lower()
	age_pop["age"] = age_pop["age_grp10"].str.replace("-"," to ").str.replace("80+","over 80")
	age_pop["age"] = age_pop["age"].str.replace("+","")

	## Create the grouped dataframe
	df = age_pop.loc[age_pop["county"].isin(group)]
	df = df[["age","pop"]].groupby("age").sum().reset_index()

	## Compute all the pieces
	population = df["pop"].sum()
	df["fraction"] = df["pop"]/(df["pop"].sum())
	pyramid = df[["age","fraction"]].set_index("age")["fraction"]

	return population, pyramid

if __name__ == "__main__":

	## Compile all the necessary data, the testing and mortality dataset,
	## the age-breakdown of cases, the age pyramid, and the population.
	timeseries = pd.read_pickle("..\\_example_datasets\\aggregated_wdrs_linelist_september20.pkl")
	timeseries = timeseries.loc[hp["counties"]].groupby("time").sum()
	population, pyramid = GetPopulationPyramid(hp["counties"])
	age_df = pd.read_pickle("..\\_example_datasets\\age_structured_cases_wdrs_september20.pkl")
	age_df = age_df.loc[hp["counties"]].groupby("time").sum().loc["2020-03-01":]
	
	## How do you handle data at the end, where increased testing and
	## lags might be an issue?
	timeseries = timeseries.loc[:hp["tr_end"]]

	## Use the dataset to compute a testing-adjusted epicurve
	_version = 1
	if _version == 0:
		epi_curve = SplineTestingEpiCurve(timeseries,debug=False)
	elif _version == 1:
		epi_curve, _, _ = RandomWalkEpiCurve(timeseries,debug=False)

	## Reindex cases to harmonize all the different timeseries.
	time = pd.date_range(start="01-15-2020",end=timeseries.index[-1],freq="d")
	cases = timeseries["cases"].reindex(time).fillna(0)
	deaths = timeseries["deaths"].reindex(time).fillna(0)
	hosp = timeseries["hosp"].reindex(time).fillna(0)
	epi_curve = epi_curve.reindex(time).fillna(0)

	## Set up empty storage for importations
	importations = pd.Series(np.zeros(len(cases),),
							 index=cases.index,
							 name="importations")

	## Set up the transmission regression start
	tr_date = pd.to_datetime(hp["tr_start"])
	tr_start = (tr_date-time[0]).days

	## Set up a model class to store relevant parameters
	## organize model fitting. 
	model = LogNormalSEIR(S0=population,
						  D_e=4,
						  D_i=8,
						  z_t=importations.values)

	## Fit the model with RAINIER using the testing corrected
	## epi-curve
	ps = np.linspace(0.015,0.04,10)
	lnbeta = np.zeros((len(ps),len(model.time[tr_start:-5])))
	lnbeta_var = np.zeros((len(ps),len(model.time[tr_start:-5])))
	print("\nEstimating transmission rates...")
	for t in enumerate(tqdm(ps)):
		i, p = t
		lnbeta[i], lnbeta_var[i] = continuous_time_posterior(p,model,epi_curve.values,
															 tr_start,debug=False)

	## Integrate (across the uniform components of the
	## p(p|C_t) and delta function approximated p(X_t|C_t,p))
	lnbeta = lnbeta.mean(axis=0)
	lnbeta_var = lnbeta_var.mean(axis=0)

	## Compute R0 point estimates
	R0_point_est = np.exp(lnbeta)*model.S0*model.D_i
	R0_point_est_std = np.exp(lnbeta)*np.sqrt(lnbeta_var)*model.S0*model.D_i
	r0_estimates = pd.DataFrame(np.array([R0_point_est,R0_point_est_std]).T,
								columns=["r0_t","std_err"],
								index=time[tr_start:tr_start+len(R0_point_est)])
	print("\nPoint estimates for R0:")
	if hp["reff_output"] is not None:
		r0_estimates.to_pickle("..\\_outputs\\"+hp["reff_output"])
	print(r0_estimates)

	## Set sigma epsilon to variance contributions from the estimates in the
	## unabated tranmission period. Not sure if this is the right thing to do?
	sig_eps = pd.Series(np.sqrt(lnbeta_var),index=r0_estimates.index)
	sig_eps = sig_eps.reindex(time).fillna(method="ffill").fillna(sig_eps.mean())#method="bfill")
	model.sig_eps = sig_eps.values
	print("\nLog-normal mean-variance relationship:")
	print("sig_eps = {}".format(model.sig_eps))

	## Set up the beta_t scenario (forward filling with restriction)
	beta_t = r0_estimates["r0_t"]/model.S0/model.D_i
	unabated_R0 = r0_estimates["r0_t"].loc[:hp["unabated_date"]].mean()
	print("\nUnabated tranmission assumed Reff = {}".format(unabated_R0))
	unabated_beta = unabated_R0/model.S0/model.D_i
	beta_t = beta_t.reindex(time).fillna(method="ffill").fillna(unabated_beta)

	## Compute the IFR for this population based on age-structed estimates
	## from Verity et al. Compute distributions via the method in rainier.rainier
	output = GaussianProcessIFR(age_df,pyramid,ifr_table)
	for est in output:
		est["std"] = np.sqrt(est["var"])
	prior_ifr, case_based_ifr, post_ifr = output
	if hp["gp_ifr"]:
		ifr = post_ifr.copy()
	else:
		ifr = prior_ifr.copy()

	## And then ensure that backfilling is based on the prior
	ifr.loc[ifr.index[0]-pd.to_timedelta(7,unit="d")] = prior_ifr.loc[prior_ifr.index[0]].values
	ifr = ifr.reindex(time).interpolate(limit_area="inside").fillna(method="bfill").fillna(method="ffill")

	## Now you have enough to compute the mean number of deaths in the
	## model conditional on z_t. Start by deciding where you want 
	## importation pulses.
	pulse_dates = [pd.to_datetime(d) for d in hp["pulses"]]
	pulse_indices = [(d-time[0]).days for d in pulse_dates]

	## Fit the mortality pulses by non-linear optimization of the
	## mean number of deaths.
	cost_function = lambda x: DeathRSS(x,model,beta_t.values,pulse_indices,deaths,
									   mean_ifr=ifr["mean"].values[:-1]/100)
	result = minimize(cost_function,
					  x0=5*np.ones((len(pulse_indices),)),
					  method="L-BFGS-B",
					  bounds=len(pulse_indices)*[(0,None)])
	print("\nImportation pulse optimization result:")
	print(result)

	## Summarize accordingly, and reset the importations in
	## the model class
	print("\nInferred pulses:")
	pulse_sizes = result["x"]
	pulse_std = np.sqrt(np.diag(result["hess_inv"].todense()))
	for d,s,std in zip(pulse_dates,pulse_sizes,pulse_std):
		print("{}: {} +/- {}".format(d.strftime("%m/%d/%Y"),s,std))
	importations.loc[pulse_dates] = result["x"]
	model.z_t = importations.values

	## Finally, sample the model starting by adjusting to the time range
	## for plotting.
	plot_time = pd.date_range(start=time[0],end=hp["plot_time_end"],freq="d")
	ticks = pd.date_range(plot_time[0],plot_time[-1],freq="SMS")
	tick_labels = [str(t.month)+"/"+str(t.day) for t in ticks]
	
	## Adjust all the model inputs to 
	importations = importations.reindex(plot_time).fillna(0)
	sig_eps = sig_eps.reindex(plot_time).fillna(method="ffill").fillna(method="bfill")
	beta_t = beta_t.reindex(plot_time).fillna(method="ffill").fillna(unabated_beta)
	ifr = ifr.reindex(plot_time).fillna(method="ffill")

	## Sample the model
	samples = model.sample(beta_t.values,
						   sig_eps=sig_eps.values,
						   z_t=importations.values)

	# Use beta binomial over time to approximate reporting
	## rates
	print("\nStarting reporting estimation...")
	i_samples = samples[:,2,tr_start:len(cases)]
	rr_samples = (cases.values[tr_start:]+1)/(i_samples+2)
	rr_t = rr_samples.mean(axis=0)
	rr_t_var = rr_samples.var(axis=0)
	rr_t_low = np.percentile(rr_samples,25.,axis=0)
	rr_t_high = np.percentile(rr_samples,75.,axis=0)

	## Fit a regression model to this data
	regime_changes = [pd.to_datetime(d) for d in hp["reporting_regimes"]]
	X = [(time <= regime_changes[0]).astype(np.float64)]
	for r0, r1 in zip(regime_changes[:-1],regime_changes[1:]):
		X.append(((time > r0) & (time <= r1)).astype(np.float64))
	X.append((time > regime_changes[-1]).astype(np.float64))
	X.append((time.weekday.isin({5,6})).astype(np.float64))
	X = np.array(X).T
	p, p_var, _ = WeightedLeastSquares(X[tr_start:],rr_t,
									   standardize=False)
	rr = np.dot(X,p)
	rr_std = np.sqrt(np.diag(np.dot(X,np.dot(p_var,X.T))))
	X_output = np.eye(len(regime_changes)+2)
	rr_o = np.dot(X_output,p)
	rr_o_std = np.sqrt(np.diag(np.dot(X_output,np.dot(p_var,X_output.T))))
	print("Reporting rate estimate before {} = {} +/- {}".format(regime_changes[0].strftime("%m/%d"),rr_o[0],rr_o_std[0]))
	for r0, r1,i in zip(regime_changes[:-1],regime_changes[1:],range(len(regime_changes)-1)):
		print("Between {} and {} = {} +/- {}".format(r0.strftime("%m/%d"),r1.strftime("%m/%d"),rr_o[i+1],rr_o_std[i+1]))
	print("Reporting rate estimate after {} = {} +/- {}".format(regime_changes[-1].strftime("%m/%d"),rr_o[-2],rr_o_std[-2]))
	print("Weekend effect = {} +/- {}".format(rr_o[-1],rr_o_std[-1]))

	## Sample the model for cases, first by forward filling the reporting rate
	## if needed.
	rr = pd.Series(rr,index=cases.index).reindex(plot_time).fillna(method="ffill")
	mask = (rr.index > time[-1]) & (rr.index.weekday.isin({5,6}))
	rr.loc[mask] = rr.loc[mask] + p[-1]
	case_samples = np.random.binomial(np.round(samples[:,2,:]).astype(int),
									  p=np.clip(rr.values,0,None))

	## Sample mortality
	ifr_samples = np.random.normal(ifr["mean"].values,
								   np.sqrt(ifr["var"].values),
								   size=((len(samples),len(ifr))))/100.
	ifr_samples = np.clip(ifr_samples,0,None)
	delay_samples = np.exp(np.random.normal(2.8329,0.42,size=(len(samples),)))
	delay_samples = np.clip(delay_samples,None,samples.shape[2]-1)
	destined_deaths, deaths_occured = SampleMortality(model,samples,
													  ifr_samples[:,:-1],
													  np.round(delay_samples).astype(int))

	## Compute active infections (and describe)
	prevalence = pd.DataFrame((samples[:,1,:] + samples[:,2,:]).T/population,
							   index=plot_time).T
	if hp["prev_output"] is not None:
		prevalence.to_pickle("..\\_outputs\\"+hp["prev_output"])
	print("\nPrevalence:")
	print(prevalence[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))
	attack_rate = pd.DataFrame((model.S0 - samples[:,0,:]).T/population,
							   index=plot_time).T
	print("\nAttack rate:")
	print(attack_rate[time[-1]].describe(percentiles=[0.025,0.25,0.5,0.75,0.975]))

	## Compute the cumulative reporting rate
	total_cases = timeseries["cases"].sum()
	cum_rr_samples = 100*total_cases/attack_rate[time[-1]]/population
	print("\nCumulative reporting rate:")
	cum_rr = cum_rr_samples.describe(percentiles=[0.025,0.25,0.5,0.75,0.975]) 
	print(cum_rr)

	## Sample hospitalizations using the same pattern we do for mortality
	## First by calculating the time-varying IHR
	prior_ihr, case_based_ihr, post_ihr = GaussianProcessIHR(age_df,pyramid,ihr_table)
	if hp["gp_ifr"]:
		ihr = post_ihr.copy()
	else:
		ihr = prior_ihr.copy()
	ihr.loc[ihr.index[0]-pd.to_timedelta(7,unit="d")] = prior_ihr.loc[prior_ihr.index[0]].values
	ihr = ihr.reindex(plot_time).interpolate(limit_area="inside").fillna(method="bfill").fillna(method="ffill")

	## Calculate an IHR scale factor by fitting to observed hospitalizations
	mean_hosp = MeanHosp(model,beta_t.values,importations.values,sig_eps.values,
						 mean_ihr=ihr["mean"].values[:-1]/100)[:len(hosp)]
	ihr_scale_factor = np.sum(mean_hosp*hosp.values)/np.sum(mean_hosp*mean_hosp)
	print("\nScale factor for hospitalization = {}".format(ihr_scale_factor))
	ihr["mean"] *= ihr_scale_factor
	ihr["var"] *= ihr_scale_factor**2
	ihr["std"] = np.sqrt(ihr["var"])

	## Then sample
	ihr_samples = np.random.normal(ihr["mean"].values,
								   np.sqrt(ihr["var"].values),
								   size=((len(samples),len(ifr))))/100.
	delay_samples = np.random.normal(4+2.1,2.65,size=(len(samples),))
	delay_samples = np.clip(delay_samples,0,samples.shape[2]-1)
	destined_hospital, hospital_occured = SampleMortality(model,samples,
														  ihr_samples[:,:-1],
														  np.round(delay_samples).astype(int))

	## If specified, output a dataframe with all
	## the key model pieces (beta, sig_eps, importations, severity
	## rates) needed to make cross-model comparisons, forecasts, etc.
	if hp["model_pickle"] is not None:
		model_pickle = pd.concat([beta_t.rename("beta_t"),
								  sig_eps.rename("sig_eps"),
								  importations.rename("z_t"),
								  rr.rename("rr")],axis=1)
		ifr.columns = ["ifr_"+c for c in ifr.columns]
		ihr.columns = ["ihr_"+c for c in ihr.columns]
		model_pickle = pd.concat([model_pickle,
								  ifr[["ifr_mean","ifr_var"]],
								  ihr[["ihr_mean","ihr_var"]]],axis=1)
		model_pickle = model_pickle.loc[:hp["tr_end"]]
		model_pickle.to_pickle("..\\_models\\"+hp["model_pickle"])
		print("\nSerialized model dataframe = ")
		print(model_pickle)
		sys.exit()

	########################################################################################
	#### Plotting.
	###########
	## Figure 1, Reff
	fig, axes = plt.subplots(figsize=(12,6.5))
	axes_setup(axes)
	axes.fill_between(r0_estimates.index,
					  r0_estimates["r0_t"].values-2.*r0_estimates["std_err"].values,
					  r0_estimates["r0_t"].values+2.*r0_estimates["std_err"].values,
					  facecolor="#F4CC70",edgecolor="None",alpha=0.3)#,label=r"95% confidence interval")
	axes.plot(r0_estimates["r0_t"],lw=3,color="#DE7A22",label=r"King County R$_e$ estimates"+" based\non tests and hospitalizations")
	if hp["reff_old"] is not None:
		old = pd.read_pickle("..\\_outputs\\"+hp["reff_old"])
		axes.plot(old["r0_t"]-2.*old["std_err"],lw=2,ls="dashed",color="#6AB197")
		axes.plot(old["r0_t"]+2.*old["std_err"],lw=2,ls="dashed",color="#6AB197")
		axes.plot(old["r0_t"],color="#20948B",lw=2,label=r"R$_e$ estimates"+" based\non last week's data")
	axes.set(ylim=(0.,4))
	axes.axhline(1.0,c="grey",lw=2,ls="dashed")#,label=r"Threshold for declining transmission, R$_{e}=1$")
	axes.text(axes.get_xlim()[0],0.95,
			  "Threshold for declining\ntransmission,"+ r" R$_{e}=1$",
			  horizontalalignment="left",verticalalignment="top",
			  fontsize=16,color="grey")
	legend = axes.legend(loc=2,frameon=True,fontsize=18)
	legend.get_frame().set_linewidth(0.0)
	axes.set_ylabel(r"Effective reproductive number (R$_{e}$)")
	r0_ticks = pd.date_range(r0_estimates.index[0],r0_estimates.index[-1],freq="W-SUN")[::2]
	r0_tick_labels = [str(t.month)+"/"+str(t.day) for t in r0_ticks]
	axes.set_xticks(r0_ticks)
	axes.set_xticklabels(r0_tick_labels)
	fig.tight_layout()
	## Add a data inset for context
	inset = fig.add_axes([0.64,0.64,0.3,0.28])
	axes_setup(inset)
	inset.grid(color="grey",alpha=0.2)
	inset.plot(timeseries["hosp"].loc["2020-02-15":hp["tr_end"]],color="k")
	inset.set_xticks(ticks[1::2])
	inset.set_xticklabels([t.strftime("%b").replace("Feb","") for t in ticks[1::2]])
	inset.set_ylabel("Hospitalizations")
	fig.savefig("..\\_plots\\example_r0.png")

	## Figure 2: IFR+IHR over time
	fig, axes = plt.subplots(2,1,sharex=True,figsize=(12,9))
	for ax in axes:
		axes_setup(ax)
		ax.grid(color="grey",alpha=0.2)
	cmap = plt.get_cmap("magma")
	colors = [cmap(x) for x in np.linspace(0.02,0.98,len(age_df.columns))]
	axes[0].fill_between(case_based_ifr.index,
					  case_based_ifr["mean"].values-2.*case_based_ifr["std"].values,
					  case_based_ifr["mean"].values+2.*case_based_ifr["std"].values,
					  facecolor=colors[2],edgecolor="None",alpha=0.3,zorder=3,label="Literature+Case based")
	axes[0].plot(case_based_ifr["mean"],color=colors[3],lw=2,zorder=4)
	axes[0].fill_between(prior_ifr.index,
					  prior_ifr["mean"].values-2.*prior_ifr["std"].values,
					  prior_ifr["mean"].values+2.*prior_ifr["std"].values,
					  facecolor=colors[0],edgecolor="None",alpha=0.3,zorder=1,label="Literature+Population based")
	axes[0].plot(prior_ifr["mean"],color=colors[1],lw=2,zorder=2)
	axes[0].fill_between(post_ifr.index,
					  post_ifr["mean"].values-2.*post_ifr["std"].values,
					  post_ifr["mean"].values+2.*post_ifr["std"].values,
					  facecolor=colors[-3],edgecolor="None",alpha=0.9,zorder=5)
	axes[0].plot(post_ifr["mean"],c=colors[-4],lw=4,label="Overall IFR estimate",zorder=6)
	axes[0].set_ylabel("Infection-fatality-ratio (IFR %)")
	axes[0].set_ylim((None,4))
	axes[0].set_xticks(ticks[3:])
	axes[0].set_xticklabels(tick_labels[3:])
	axes[0].legend(loc=1,fontsize=20)

	## Panel 2: IHR
	for s in [case_based_ihr,prior_ihr,post_ihr]:
		s["mean"] *= ihr_scale_factor
		s["var"] *= ihr_scale_factor**2
		s["std"] = np.sqrt(s["var"])
	axes[1].fill_between(case_based_ihr.index,
					  case_based_ihr["mean"].values-2.*case_based_ihr["std"].values,
					  case_based_ihr["mean"].values+2.*case_based_ihr["std"].values,
					  facecolor=colors[2],edgecolor="None",alpha=0.3,zorder=3,label="Literature+Case based")
	axes[1].plot(case_based_ihr["mean"],color=colors[3],lw=2,zorder=4)
	axes[1].fill_between(prior_ihr.index,
					  prior_ihr["mean"].values-2.*prior_ihr["std"].values,
					  prior_ihr["mean"].values+2.*prior_ihr["std"].values,
					  facecolor=colors[0],edgecolor="None",alpha=0.3,zorder=1,label="Literature+Population based")
	axes[1].plot(prior_ihr["mean"],color=colors[1],lw=2,zorder=2)
	axes[1].fill_between(post_ihr.index,
					  post_ihr["mean"].values-2.*post_ihr["std"].values,
					  post_ihr["mean"].values+2.*post_ihr["std"].values,
					  facecolor=colors[-3],edgecolor="None",alpha=0.9,zorder=5)
	axes[1].plot(post_ihr["mean"],c=colors[-4],lw=4,label="Overall IHR estimate",zorder=6)
	axes[1].set_ylabel("Infection-hospitalization-ratio (IHR %)")
	axes[1].set_xticks(ticks[3:])
	axes[1].set_xticklabels(tick_labels[3:])
	axes[1].legend(loc=1,fontsize=20)
	fig.tight_layout()
	fig.savefig("..\\_plots\\example_rates.png")
	
	## Figure 3: Model fit
	fig, axes = plt.subplots(3,1,figsize=(14,15))
	for ax in axes:
		axes_setup(ax)

	## Panel 1: Cases
	sd_l0, sd_h0, sd_l1, sd_h1, sd_l2, sd_h2 = low_mid_high(case_samples)
	axes[0].fill_between(plot_time,sd_l0,sd_h0,color="#2C7873",alpha=0.1,zorder=1)
	axes[0].fill_between(plot_time,sd_l1,sd_h1,color="#2C7873",alpha=0.2,zorder=2)
	axes[0].fill_between(plot_time,sd_l2,sd_h2,color="#2C7873",alpha=0.8,zorder=3,label="Cases in the transmission model")
	axes[0].plot(plot_time,case_samples.mean(axis=0),lw=2,color="#2C7873")
	axes[0].plot(cases.loc["2020-02-26":],
			  ls="None",marker="o",markersize=7,
			  markeredgecolor="k",markerfacecolor="k",markeredgewidth=1,zorder=4,
			  label="Daily WDRS positive COVID-19 tests")

	## Details
	axes[0].legend(loc=2,frameon=False,fontsize=18)
	axes[0].set_ylabel("COVID-19 cases")
	axes[0].set_xticks(ticks)
	axes[0].set_xticklabels(tick_labels)

	## Panel 2: Hospitalizations
	l0, h0, l1, h1, l2, h2 = low_mid_high(hospital_occured)
	axes[1].fill_between(plot_time[1:],l0,h0,color="#F18D9E",alpha=0.1)
	axes[1].fill_between(plot_time[1:],l1,h1,color="#F18D9E",alpha=0.3)
	axes[1].fill_between(plot_time[1:],l2,h2,color="#F18D9E",alpha=0.6,label="Severe infections in the transmission model")
	axes[1].plot(plot_time[1:],hospital_occured.mean(axis=0),lw=2,color="#F18D9E")
	axes[1].plot(timeseries.loc[timeseries.loc[timeseries["hosp"]!=0].index[0]:,"hosp"],
			  ls="None",color="k",marker="o",markersize=7,label="Daily COVID-19 hospitalizations reported to the WDRS")
	axes[1].legend(loc=2,frameon=False,fontsize=18)
	axes[1].set_ylabel("COVID-19 hospital admissions")
	axes[1].set_xticks(ticks)
	axes[1].set_xticklabels(tick_labels)
	
	## Panel 2: Deaths
	l0, h0, l1, h1, l2, h2 = low_mid_high(deaths_occured)
	axes[2].fill_between(plot_time[1:],l0,h0,color="#F98866",alpha=0.1)
	axes[2].fill_between(plot_time[1:],l1,h1,color="#F98866",alpha=0.3)
	axes[2].fill_between(plot_time[1:],l2,h2,color="#F98866",alpha=0.6,label="Deaths in the transmission model")
	axes[2].plot(plot_time[1:],deaths_occured.mean(axis=0),lw=2,color="#F98866")
	axes[2].plot(timeseries.loc[timeseries.loc[timeseries["deaths"]!=0].index[0]:,"deaths"],
			  ls="None",color="k",marker="o",markersize=7,label="Daily COVID-19 deaths reported to the WDRS")
	axes[2].legend(loc=2,frameon=False,fontsize=18)
	axes[2].set_ylabel("COVID-19 deaths")
	axes[2].set_xticks(ticks)
	axes[2].set_xticklabels(tick_labels)
	
	## Save the multipanel
	fig.tight_layout()
	fig.savefig("..\\_plots\\example_fit.png")

	## Figure 4: Prevalence
	fig, axes = plt.subplots(2,1,figsize=(12,10))
	for ax in axes:
		axes_setup(ax)

	## Plot prevalence over time
	l0, h0, l1, h1, l2, h2 = low_mid_high(100*prevalence.values)
	axes[0].fill_between(plot_time,l0,h0,color="#7F152E",alpha=0.1)
	axes[0].fill_between(plot_time,l1,h1,color="#7F152E",alpha=0.3)
	axes[0].fill_between(plot_time,l2,h2,color="#7F152E",alpha=0.8,
						 label="Estimated prevalence of active infections\nusing the transmission model")
	axes[0].plot(plot_time,(100*prevalence.values).mean(axis=0),color="#7F152E",lw=2)
	axes[0].legend(loc=1,frameon=False,fontsize=18)
	axes[0].set_ylabel(r"COVID-19 prevalence (%)")
	axes[0].set_xticks(ticks)
	axes[0].set_xticklabels(tick_labels)
	axes[0].set_xlim(("2020-01-15",None))

	## Plot cummulative incidence/attack rate over time
	l0, h0, l1, h1, l2, h2 = low_mid_high(100*attack_rate.values)
	axes[1].fill_between(plot_time,l0,h0,color="#4D648D",alpha=0.1)
	axes[1].fill_between(plot_time,l1,h1,color="#4D648D",alpha=0.3)
	axes[1].fill_between(plot_time,l2,h2,color="#4D648D",alpha=0.8,
						 label="Estimated cumulative incidence using the transmission model")
	axes[1].plot(plot_time,(100*attack_rate.values).mean(axis=0),color="#4D648D",lw=2)
	axes[1].legend(loc=2,frameon=False,fontsize=18)
	axes[1].set_ylabel(r"COVID-19 cumulative incidence (%)")
	axes[1].set_xticks(ticks)
	axes[1].set_xticklabels(tick_labels)
	axes[1].set_xlim(("2020-01-15",None))
	
	## Finalize the main figure
	fig.tight_layout()

	## Add a reporting inset
	inset_dim = [0.105, 0.22, 0.20, 0.15]
	axes2 = fig.add_axes(inset_dim)
	axes2.spines["left"].set_visible(False)
	axes2.spines["top"].set_visible(False)
	axes2.spines["right"].set_visible(False)

	## Compute a KDE for the cumulative reporting rate
	kde = gaussian_kde(cum_rr_samples.values)
	rr = np.linspace(0,50,1000)
	hist = kde(rr)
	axes2.fill_between(rr,0,hist,color="grey",alpha=0.4)
	axes2.plot(rr,hist,lw=3,color="k")
	axes2.set_ylim((0,None))
	axes2.set_yticks([])
	axes2.set_xlabel("Incidence reported (%)")

	## Save the output
	fig.savefig("..\\_plots\\example_output.png")

	## Figure 5: Time dependent reporting
	weekly_cases = cases.resample("W-SUN").sum().loc[hp["pulses"][0]:]
	active_infections = pd.DataFrame((samples[:,1,:] + samples[:,2,:]).T,
		 						   index=plot_time)
	weekly_infections = active_infections.resample("W-SUN").mean().loc[weekly_cases.index]
	weekly_reporting_samples = 100*weekly_cases.values/(weekly_infections.T.values)

	## Make a plot
	fig, axes = plt.subplots(figsize=(12,5))
	axes_setup(axes)
	l0, h0, l1, h1, l2, h2 = low_mid_high(weekly_reporting_samples)
	avg_rr = weekly_reporting_samples.mean(axis=0)
	rr_df = pd.DataFrame(np.array([l0, h0, l1, h1, l2, h2, avg_rr]).T,
						 index=weekly_cases.index,
						 columns=["l0", "h0", "l1", "h1", "l2", "h2", "avg_rr"])
	rr_df = rr_df.iloc[:-1]
	rr_df = rr_df.reindex(cases.index).fillna(method="bfill").dropna()
	axes.fill_between(rr_df.index,rr_df["l0"].values,rr_df["h0"].values,color="#063852",alpha=0.1)
	axes.fill_between(rr_df.index,rr_df["l1"].values,rr_df["h1"].values,color="#063852",alpha=0.3)
	axes.fill_between(rr_df.index,rr_df["l2"].values,rr_df["h2"].values,color="#063852",alpha=0.6,
					  label="Weekly estimates for King County, computed\nby comparing case reports to model infections")
	axes.plot(rr_df.index,rr_df["avg_rr"].values,color="#063852",lw=2)
	axes.axvline("2020-06-05",ymax=0.6,ls="dashed",lw=2,color="#FFBB00")
	axes.text("2020-06-05",32,
			  "Free, drive-thru testing\nbecomes available",
			  horizontalalignment="right",verticalalignment="bottom",fontsize=18,
			  color="#FFBB00")
	axes.legend(loc=2,frameon=False,fontsize=18)
	axes.set_ylabel("Percent of incidence reported (%)")
	axes.set_xticks(ticks)
	axes.set_xticklabels(tick_labels)
	fig.tight_layout()
	fig.savefig("..\\_plots\\example_overallreporting.png")

	## Done
	plt.show()





