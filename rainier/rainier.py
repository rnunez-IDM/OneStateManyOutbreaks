""" rainier.py

Utilities for RAINIER (Reporting Adjusted Immuno-Naive, Infected, Exposed Regression). These
functions and classes manage analysis of case-data time series for comparison with regression
models. """
import sys
sys.path.append("..\\")
import warnings

## Standard imports
import numpy as np
import pandas as pd

## For debug/step-by-step plots
import matplotlib.pyplot as plt

## For estimating hidden state variables and
## associated uncertainty.
from rainier.splines import SmoothingSpline, SampleSpline

## For epi-curve weights
from scipy.special import gammaln, beta
from scipy.linalg import block_diag
from scipy.optimize import minimize

#### Data prep functions
def SplineTestingEpiCurve(dataset,debug=False):

	""" Create an epi curve based on fraction positive and smoothed total tests. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. Smoothing here is done
	using a smoothing spline with a 3 day prior correlation. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Compute spline smoothed total tests
	spline = SmoothingSpline(np.arange(len(total_tests)),total_tests.values,lam=((3**4)/8))
	smooth_tt = pd.Series(spline(np.arange(len(dataset))),
						  index=dataset.index) 
	smooth_tt.loc[smooth_tt<0] = 0

	## Compute the epicurve estimate
	epi_curve = fraction_positive*smooth_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,10))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=1,
					 label="WDRS COVID-19 positives")
		axes[0].plot(fraction_positive*smooth_tt,c="xkcd:red wine",lw=2,
					 label="Epidemiological curve, based on smoothed\ntests, used to estimate " +r"R$_e$")
		axes[1].plot(total_tests,c="k",ls="dashed",lw=1,
					 label="Raw total daily tests")
		axes[1].plot(smooth_tt,c="xkcd:red wine",lw=2,
					 label="Smoothed tests with a 3 day correlation\ntime, correcting for fluctuations")
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive.loc[fraction_positive.loc[fraction_positive!=0].index[0]:],
					 c="grey",lw=3,
					 label="Raw fraction positive, computed with WDRS positive\nand negative tests, declines with increased testing volume")
		for ax in axes:
			ax.legend(frameon=False,fontsize=18)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total COVID-19 tests")
		axes[2].set_ylabel("Fraction positive")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

def HybridTestingEpiCurve(dataset,
						  hybrid_date="2020-04-15",debug=False):

	""" Create an epi curve based on fraction positive and smoothed total tests, which then
	switches to step-wise at the specified date. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. Smoothing here is done
	using a smoothing spline with a 3 day prior correlation. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Compute spline smoothed total tests
	spline = SmoothingSpline(np.arange(len(total_tests)),total_tests.values,lam=((3**4)/8))
	smooth_tt = pd.Series(spline(np.arange(len(dataset))),
						  index=dataset.index) 
	smooth_tt.loc[smooth_tt<0] = 0

	## Forward fill to switch to stepwise
	smooth_tt.loc[hybrid_date:] = np.nan
	smooth_tt = smooth_tt.fillna(method="ffill")

	## Compute the epicurve estimate
	epi_curve = fraction_positive*smooth_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,10))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=1,
					 label="WDRS COVID-19 positives")
		axes[0].plot(fraction_positive*smooth_tt,c="xkcd:red wine",lw=2,
					 label="Epidemiological curve, based on smoothed\ntests, used to estimate " +r"R$_e$")
		axes[1].plot(total_tests,c="k",ls="dashed",lw=1,
					 label="Raw total daily tests")
		axes[1].plot(smooth_tt,c="xkcd:red wine",lw=2,
					 label="Smoothed tests with a 3 day correlation time,\n constant after "+hybrid_date)
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive.loc[fraction_positive.loc[fraction_positive!=0].index[0]:],
					 c="grey",lw=2,
					 label="Raw fraction positive, computed with WDRS positive\nand negative tests, declines with increased testing volume")
		for ax in axes:
			ax.legend(frameon=False,fontsize=18)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total COVID-19 tests")
		axes[2].set_ylabel("Fraction positive")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

def StepTestingEpiCurve(dataset,regimes,debug=False):

	""" Create an epi curve based on fraction positive and step-wise total tests. dataset is a 
	dataframe with a daily time index with cases and negatives as columns. total tests is set to the mean
	in windows based on regimes, a list of date-times where splits are made. """

	## Compute fraction positive
	total_tests = dataset["cases"]+dataset["negatives"]
	fraction_positive = (dataset["cases"]/total_tests).fillna(0)

	## Step tests approximation
	regime_indices = [(d-dataset.index[0]).days for d in regimes]
	step_tt = np.split(total_tests,regime_indices)
	for i,s in enumerate(step_tt):
		if i == 0:
			continue
		s.loc[:] = int(np.round(s.mean()))
	step_tt = pd.concat(step_tt,axis=0)

	## Compute the epicurve estimate
	epi_curve = fraction_positive*step_tt

	## Make a diagnostic plot if needed
	if debug:
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,10))
		axes[0].plot(dataset["cases"],c="k",ls="dashed",lw=1,label="Cases")
		axes[0].plot(epi_curve,c="xkcd:red wine",lw=1,label="Step-wise approximation")
		axes[1].plot(total_tests,c="k",lw=2,label="Total tests")
		axes[1].plot(step_tt,c="xkcd:red wine",lw=1,label="Step-function approximation")
		axes[1].set_xlim(("2020-02-01",None))
		axes[2].plot(fraction_positive,c="grey",lw=2,label="Raw fraction positive")
		for ax in axes:
			ax.legend(frameon=False,fontsize=18)
		axes[0].set_ylabel("Epi-curve")
		axes[1].set_ylabel("Total tests")
		axes[2].set_ylabel("Fraction positive from WDRS")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return epi_curve

def RandomWalkEpiCurve(timeseries,
					   correlation_time=28,debug=False):

	""" Construct an epi-curve based on a combination of testing and hospitalization data,
	with an eye towards pooling information for better estimates.

	timeseries is a data frame with cols cases, negatives, and hospitalization. 
	correlation_time: in days, for the random walk. corresponds to the time-scale on which
					  hospitalization data should be followed. """

	## Construct the spline epi curve using the function above.
	## This essentially interprets the cases as directly connected to
	## epi when corrected for weekend effects.
	spline = SplineTestingEpiCurve(timeseries,debug=False).rename("spline")
	
	## Concatenate into a single, aligned dataframe to make
	## referencing easier throughout.
	df = pd.concat([spline,timeseries["hosp"]],axis=1)
	df = df.loc[spline.loc[spline>0].index[0]:]

	## Then set up the regularization matrix for the parameters 
	## (fixed effect for the scale factor, random walk for the weights).
	T = len(df)
	D2 = np.diag(T*[-2])+np.diag((T-1)*[1],k=1)+np.diag((T-1)*[1],k=-1)
	D2[0,2] = 1
	D2[-1,-3] = 1
	lam = np.dot(D2.T,D2)*((correlation_time**4)/8.)
	
	## Set up cost function to be passed to scipy.minimize, and then
	## solve the regression problem.
	def cost(theta):
		beta = 1./(1. + np.exp(-theta))
		f = beta*df["spline"].values
		ll = np.sum((df["hosp"].values-f)**2)
		lp = np.dot(theta.T,np.dot(lam,theta))
		return ll+lp
	alpha = np.sum(spline.values*timeseries["hosp"].values)/np.sum(spline.values**2)
	x0 = np.log(alpha/(1-alpha))*np.ones((T,))
	result = minimize(cost,x0=x0,
					  options={"gtol":1e-3})
	beta = 1./(1. + np.exp(-result["x"]))
	
	## Finally, construct the fitted epi-curve
	epi_curve = (beta*df["spline"].copy()).reindex(spline.index).fillna(0)
	rw = pd.Series(beta,index=df.index,name="rw").reindex(spline.index).fillna(method="bfill")

	## Plot if debug
	if debug:
		print("\nRegression result")
		print(result)

		## Set up a figure
		fig, axes = plt.subplots(3,1,sharex=True,figsize=(12,11))
		for i, ax in enumerate(axes):
			ax.spines["left"].set_position(("axes",-0.015))
			if i != 2:
				ax.spines["left"].set_visible(False)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)

		## Plot the case data
		axes[0].plot(timeseries["cases"],ls="None",marker="o",markersize=9,alpha=0.5,
					 markeredgecolor="k",markerfacecolor="None",markeredgewidth=1,
					 lw=1,color="k",label="Daily COVID-19 cases")
		axes[0].plot(spline,lw=3,color="#8F2D56",label=r"$\tilde{C}_t$, "+"epi-curve based\non testing data only")
		axes[0].set_ylim((0,None))
		axes[0].set_yticks([])
		axes[0].legend(loc="upper left",
					   bbox_to_anchor=(-0.08,0.95),
					   fontsize=20,frameon=False)
		
		## Plot the epi curve
		axes[1].plot(timeseries["hosp"],ls="None",marker="o",markersize=9,alpha=0.5,
					 markeredgecolor="k",markerfacecolor="None",markeredgewidth=1,
					 lw=1,color="k",label="Daily hospital admissions")
		axes[1].plot(epi_curve,lw=3,color="#EDAE01",label=r"$\tilde{H}_t$, "+"epi-curve regularized\nby hospitalizations")
		axes[1].set_ylim((0,None))
		axes[1].set_yticks([])
		axes[1].legend(loc="upper left",
					   bbox_to_anchor=(-0.08,0.95),
					   fontsize=20,frameon=False)
		
		## Set up the middle panel
		axes[2].fill_between(rw.index,0,rw.values,color="#662E1C",alpha=0.2)
		axes[2].plot(rw,c="#662E1C",lw=4)
		axes[2].set_ylim((0,None))
		axes[2].set_ylabel(r"Transformed random walk, $f(\mu_t^*)$")

		## Finish up details
		ticks = pd.date_range(timeseries.index[0],timeseries.index[-1],freq="MS")
		tick_labels = [t.strftime("%b") for t in ticks]
		axes[1].set_xticks(ticks)
		axes[1].set_xticklabels(tick_labels)
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug_blended.png")

		## Testing figure for reference
		timeseries["tests"] = timeseries["cases"]+timeseries["negatives"]
		fig, axes = plt.subplots(figsize=(12,5))
		axes.spines["left"].set_position(("axes",-0.025))
		axes.spines["top"].set_visible(False)
		axes.spines["right"].set_visible(False)
		axes.plot(timeseries["tests"],color="k",lw=1,ls="dashed")
		axes.plot(timeseries["tests"].rolling(7).mean(),color="#505160",lw=3)
		axes.set_ylabel("Daily COVID-19 tests")
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug_tests.png")

		plt.show()
		sys.exit()

	return epi_curve, rw, spline

#### Time-varying IFR
##############################################################################
def GaussianProcessIFR(age_structured_cases,pyramid,ifr,
					   num_samples=10000,vectorized=True):

	""" Compute age-trend adjusted IFR over time. age_structured_cases is a weekly
	dataframe, with age-bins for columns. IFR and pyramid are dataframes with IFR estimates
	by age bin and population fraction by age bin respectively. 

	This function convolves an age-pyramid based prior with weekly case-based trends to get
	a weekly IFR estimate with uncertainty that responds to definitive transient changes in 
	the ages of people being infected. 

	Output is a dataframe of weekly IFR estimates and variance, to be resamples and interpolated
	for IFR-based initial condition estimation and mortality fitting. """

	## Get the length of time series and number of age bins for
	## reference throughout.
	T = len(age_structured_cases)
	K = len(ifr)

	## Compute a set of IFR samples in each age bin, assuming
	## the IFR is uniformly distributed on the 95% CI.
	ifr_samples = 100*np.random.uniform(low=ifr["low"].values,
										high=ifr["high"].values,
										size=(num_samples,len(ifr)))

	## Compute the time invariate, demographic based estimate
	time_invariant = np.dot(ifr_samples,pyramid.values)
	#prior_mean = 1.5*np.ones((len(df),)) #time_invariant.mean()*np.ones((len(df),))
	#prior_var = 0.49*np.ones((len(df),)) #time_invariant.var()*np.ones((len(df),))
	prior_mean = time_invariant.mean()*np.ones((T,))
	prior_var = time_invariant.var()*np.ones((T,))
	prior = pd.DataFrame(np.array([prior_mean,prior_var]).T,
						 index=age_structured_cases.index,
						 columns=["mean","var"])

	## Sample a dirichlet distribution to compute the case-based
	## time varying estimate. This is done either with a (less interpretable)
	## vectorized option, or via a loop over time.
	alpha = age_structured_cases.copy()+1
	if vectorized:
		dist_samples = np.random.standard_gamma(alpha.values,size=(num_samples,T,K))
		dist_samples = dist_samples/dist_samples.sum(axis=-1,keepdims=True)
		avg_ifr_samples = (dist_samples*(ifr_samples[:,np.newaxis,:])).sum(axis=-1)
		case_based = pd.DataFrame(np.array([avg_ifr_samples.mean(axis=0),avg_ifr_samples.var(axis=0)]).T,
								  index=alpha.index,
								  columns=["mean","var"])
	else:
		case_based = []
		for t, a in alpha.iterrows():
			dist_samples = np.random.dirichlet(a.values,size=(num_samples,))
			avg_ifr_samples = np.sum(ifr_samples*dist_samples,axis=1)
			case_based.append([avg_ifr_samples.mean(),avg_ifr_samples.var()])
		case_based = pd.DataFrame(case_based,
								  index=alpha.index,
								  columns=["mean","var"])

	## Compute a posterior estimate, using the population based
	## estimate as a regularizing prior on the case-based estimate.
	post_mean = (prior["var"]*case_based["mean"]+case_based["var"]*prior["mean"])/(case_based["var"]+prior["var"])
	post_var = (case_based["var"]*prior["var"])/(case_based["var"]+prior["var"])
	post = pd.concat([post_mean.rename("mean"),post_var.rename("var")],axis=1)
	
	return prior, case_based, post

#### Time-varying IHR
##############################################################################
def GaussianProcessIHR(age_structured_cases,pyramid,ihr,
					   num_samples=10000,vectorized=True):

	""" Compute age-trend adjusted IHR over time. age_structured_cases is a weekly
	dataframe, with age-bins for columns. IHR and pyramid are dataframes with IHR estimates
	by age bin and population fraction by age bin respectively. 

	This function convolves an age-pyramid based prior with weekly case-based trends to get
	a weekly IHR estimate with uncertainty that responds to definitive transient changes in 
	the ages of people being infected. 

	Output is a dataframe of weekly IHR estimates and variance, to be resamples and interpolated
	for IHR-based initial condition estimation and mortality fitting. """

	## Get the length of time series and number of age bins for
	## reference throughout.
	T = len(age_structured_cases)
	K = len(ihr)

	## Compute a set of IFR samples in each age bin, assuming
	## the IFR is uniformly distributed on the 95% CI.
	ihr_samples = 100*np.random.uniform(low=ihr["low"].values,
										high=ihr["high"].values,
										size=(num_samples,len(ihr)))

	## Compute the time invariate, demographic based estimate
	time_invariant = np.dot(ihr_samples,pyramid.values)
	prior_mean = time_invariant.mean()*np.ones((T,))
	prior_var = time_invariant.var()*np.ones((T,))
	prior = pd.DataFrame(np.array([prior_mean,prior_var]).T,
						 index=age_structured_cases.index,
						 columns=["mean","var"])

	## Sample a dirichlet distribution to compute the case-based
	## time varying estimate. This is done either with a (less interpretable)
	## vectorized option, or via a loop over time.
	alpha = age_structured_cases.copy()+1
	if vectorized:
		dist_samples = np.random.standard_gamma(alpha.values,size=(num_samples,T,K))
		dist_samples = dist_samples/dist_samples.sum(axis=-1,keepdims=True)
		avg_ihr_samples = (dist_samples*(ihr_samples[:,np.newaxis,:])).sum(axis=-1)
		case_based = pd.DataFrame(np.array([avg_ihr_samples.mean(axis=0),avg_ihr_samples.var(axis=0)]).T,
								  index=alpha.index,
								  columns=["mean","var"])
	else:
		case_based = []
		for t, a in alpha.iterrows():
			dist_samples = np.random.dirichlet(a.values,size=(num_samples,))
			avg_ihr_samples = np.sum(ihr_samples*dist_samples,axis=1)
			case_based.append([avg_ihr_samples.mean(),avg_ihr_samples.var()])
		case_based = pd.DataFrame(case_based,
								  index=alpha.index,
								  columns=["mean","var"])

	## Compute a posterior estimate, using the population based
	## estimate as a regularizing prior on the case-based estimate.
	post_mean = (prior["var"]*case_based["mean"]+case_based["var"]*prior["mean"])/(case_based["var"]+prior["var"])
	post_var = (case_based["var"]*prior["var"])/(case_based["var"]+prior["var"])
	post = pd.concat([post_mean.rename("mean"),post_var.rename("var")],axis=1)
	
	return prior, case_based, post

#### Distribution functions
##############################################################################
def continuous_time_posterior(p,model,cases,tr_start,debug=False):

	""" This is the reporting conditional posterior on log_beta_t using a continuous time
	based approximation to the population in the exposed compartment. """


	## Start by constructing coarse estimates of I_t and E_t during the
	## testing period.
	coarse_I = cases/p
	coarse_E = (model.D_e/model.D_i)*coarse_I[model.D_e:]

	## Smooth the noise based on characteristic variation time
	## in each compartment.
	spline_e = SmoothingSpline(model.time[:len(model.time)-model.D_e],coarse_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time,coarse_I,
							   lam=(model.D_i**4)/8)

	## Evaluate E and I plus covariance
	Ihat, Icov = spline_i(model.time,cov=True)
	Ehat, Ecov = spline_e(model.time,cov=True)

	## For the new exposures, we need a difference matrix (that should 
	## eventually be precomputed and stored in a RAINIER class?)
	e_diff_matrix = np.diag((-1.+(1./model.D_e))*np.ones((len(model.time)-1,)))\
				  + np.diag(np.ones((len(model.time)-2,)),k=1)
	e_diff_matrix = np.hstack([e_diff_matrix,np.zeros((len(model.time)-1,1))])
	e_diff_matrix[-1,-1] = 1
	delta_E = np.dot(e_diff_matrix,Ehat)
	delta_E_cov = np.dot(e_diff_matrix,np.dot(Ecov,e_diff_matrix.T))

	## Now, use a cumulative sum matrix (also to be precomputed in a 
	## RAINIER class and stored) to propogate uncertainty to s
	cum_sum = np.tril(np.ones((len(model.time)-1,len(model.time)-1)))
	Shat = model.S0 - np.dot(cum_sum,delta_E)
	Scov = np.dot(cum_sum,np.dot(delta_E_cov,cum_sum.T))

	## Finally, compute Yhat (the mean log transmission rate)
	Yhat = np.log(delta_E[tr_start:-model.D_e]) \
		   - np.log(Shat[tr_start-1:-model.D_e-1]) \
		   - np.log(Ihat[tr_start:-model.D_e-1]+model.z_t[tr_start:-model.D_e-1])

	## And approximate uncertainty - this is done without covariance
	## terms because the operating assumption of the model is a Markov
	## property (i.e. log(S_{t-1}) is applied to that point estimate only,
	## and the marginal distributions of the point estimates give you just
	## the diagonal elements)
	Yvar = np.diag(delta_E_cov)[tr_start:-model.D_e]/(delta_E[tr_start:-model.D_e]**2)\
		   + np.diag(Scov)[tr_start-1:-model.D_e-1]/(Shat[tr_start-1:-model.D_e-1]**2)\
		   + np.diag(Icov)[tr_start:-model.D_e-1]/(Ihat[tr_start:-model.D_e-1]**2)

	if debug:

		fig, axes = plt.subplots(4,1,sharex=True,figsize=(12,13))

		## Step 1: I_t
		I_std = np.sqrt(np.diag(Icov))
		axes[0].fill_between(model.time,Ihat-2.*I_std,Ihat+2.*I_std,color="#FFBB00",alpha=0.25)
		axes[0].plot(model.time,coarse_I,c="k",ls="dashed",label="Case data scaled by the reporting rate")
		axes[0].plot(model.time,Ihat,c="#FFBB00",label="Estimated infectious population")
		axes[0].axvline(tr_start,c="k")
		axes[0].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 2: E_t
		E_std = np.sqrt(np.diag(Ecov))
		delta_E_std = np.sqrt(np.diag(delta_E_cov))
		axes[1].fill_between(model.time,Ehat-2.*E_std,Ehat+2.*E_std,color="#FB6542",alpha=0.25)
		axes[1].plot(model.time[:len(model.time)-model.D_e],coarse_E,c="grey",ls="dashed",
					 label="Scaled case data shifted by the latent period")
		axes[1].plot(model.time,Ehat,c="#FB6542",label="Estimated exposed population")
		axes[1].fill_between(model.time[:-1],delta_E-2.*delta_E_std,delta_E+2.*delta_E_std,color="k",alpha=0.25)
		axes[1].plot(model.time[:-1],delta_E,c="k",label="Estimated new exposures per day")
		axes[1].axvline(tr_start,c="k")
		axes[1].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 3: S panel
		S_std = np.sqrt(np.diag(Scov))
		axes[2].fill_between(model.time[1:],Shat-2.*S_std,Shat+2.*S_std,color="#375E97",alpha=0.25)
		axes[2].plot(model.time[1:],Shat,color="#375E97",label="Estimated susceptible population\nbased on daily new exposures")
		axes[2].axvline(tr_start,c="k")
		axes[2].axvline(len(model.time)-model.D_e-1,c="k")

		## Step 4: log transmission rate panel
		Y_std = np.sqrt(Yvar)
		axes[3].fill_between(model.time[tr_start:-model.D_e-1],Yhat-2.*Y_std,Yhat+2.*Y_std,color="#3F681C",alpha=0.25)
		axes[3].plot(model.time[tr_start:-model.D_e-1],Yhat,c="#3F681C",
					 label="Estimated log transmission rate "+r"$(\log\beta)$"+"\ncomputed via hidden state estimates above")
		axes[3].axvline(tr_start,c="k")
		axes[3].axvline(len(model.time)-model.D_e-1,c="k")

		## Some labels
		for i, ax in enumerate(axes):
			ax.legend(frameon=False,fontsize=18)
			ax.set_yticks([])
			ax.set_ylabel("Step {}".format(i+1),fontweight="bold")
		axes[-1].set_xlabel("Model time step")

		## Annotations for the vertical bars
		axis_to_data = axes[3].transAxes + axes[3].transData.inverted()
		axes[3].text(tr_start,axis_to_data.transform((0,0.5))[1],
					 "Analysis period begins\n",fontsize=14,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=90)
		axes[3].text(len(model.time)-model.D_e-1,axis_to_data.transform((0,0.5))[1],
					 "Analysis period ends\n",fontsize=14,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=-90)
		## Done
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return Yhat, Yvar

def continuous_time_posterior_sampler(p,model,cases,num_spline_samples,tr_start,debug=False):

	""" This is the reporting conditional posterior on log_beta_t using a continuous time
	based approximation to the population in the exposed compartment. Here, uncertainty is quantified
	via bootstrap samples. """

	## Start by constructing coarse estimates of I_t and E_t during the
	## testing period.
	coarse_I = cases/p
	coarse_E = (model.D_e/model.D_i)*coarse_I[model.D_e:]

	## Smooth the noise based on characteristic variation time
	## in each compartment.
	spline_e = SmoothingSpline(model.time[:len(model.time)-model.D_e],coarse_E,
							   lam=(model.D_e**4)/8)
	spline_i = SmoothingSpline(model.time,coarse_I,
							   lam=(model.D_i**4)/8)

	## Compute smoothed estimates
	smooth_I = SampleSpline(spline_i,model.time,num_samples=num_spline_samples)
	smooth_E = SampleSpline(spline_e,model.time,num_samples=num_spline_samples)
	new_exposures_t = smooth_E[:,1:] - (1.-(1./model.D_e))*smooth_E[:,:-1]
	smooth_S = model.S0 - np.cumsum(new_exposures_t,axis=0)

	## Create the regression response vector by comparing
	## S, E, and I estimates over time
	Y = np.log(smooth_E[:,tr_start+1:-model.D_e] - (1.-(1./model.D_e))*smooth_E[:,tr_start:-model.D_e-1]) \
		- np.log(smooth_S[:,tr_start-1:-model.D_e-1]) \
		- np.log(smooth_I[:,tr_start:-model.D_e-1]+model.z_t[np.newaxis,tr_start:-model.D_e-1])

	if debug:

		## S panel
		fig, axes = plt.subplots(4,1,sharex=True,figsize=(18,17))
		m = smooth_S.mean(axis=0)
		l = np.percentile(smooth_S,25.,axis=0)
		h = np.percentile(smooth_S,75.,axis=0)
		axes[2].fill_between(model.time[1:],l,h,color="#375E97",alpha=0.25)
		axes[2].plot(model.time[1:],m,color="#375E97",label="Estimated susceptible population\nbased on daily new exposures")
		axes[2].axvline(tr_start,c="k")
		axes[2].axvline(len(model.time)-model.D_e-1,c="k")

		## log transmission rate panel
		m = Y.mean(axis=0)
		l = np.percentile(Y,25.,axis=0)
		h = np.percentile(Y,75.,axis=0)
		axes[3].fill_between(model.time[tr_start:-model.D_e-1],l,h,color="#3F681C",alpha=0.25)
		axes[3].plot(model.time[tr_start:-model.D_e-1],m,c="#3F681C",
					 label="Estimated log transmission rate "+r"$(\log\beta)$"+"\ncomputed via hidden state estimates above")
		axes[3].axvline(tr_start,c="k")
		axes[3].axvline(len(model.time)-model.D_e-1,c="k")

		## I plot and E plot
		m_i = smooth_I.mean(axis=0)
		l_i = np.percentile(smooth_I,25.,axis=0)
		h_i = np.percentile(smooth_I,75.,axis=0)		
		m_e = smooth_E.mean(axis=0)
		l_e = np.percentile(smooth_E,25.,axis=0)
		h_e = np.percentile(smooth_E,75.,axis=0)	
		m_n = new_exposures_t.mean(axis=0)
		l_n = np.percentile(new_exposures_t,25.,axis=0)
		h_n = np.percentile(new_exposures_t,75.,axis=0)	
		axes[0].fill_between(model.time,l_i,h_i,color="#FFBB00",alpha=0.25)
		axes[0].plot(model.time,coarse_I,c="k",ls="dashed",label="Case data scaled by the reporting rate")
		axes[0].plot(model.time,m_i,c="#FFBB00",label="Estimated infectious population")
		axes[0].axvline(tr_start,c="k")
		axes[0].axvline(len(model.time)-model.D_e-1,c="k")
		axes[1].fill_between(model.time,l_e,h_e,color="#FB6542",alpha=0.25)
		axes[1].plot(model.time[:len(model.time)-model.D_e],coarse_E,c="grey",ls="dashed",
					 label="Scaled case data shifted by the latent period")
		axes[1].plot(model.time,m_e,c="#FB6542",label="Estimated exposed population")
		axes[1].fill_between(model.time[:-1],l_n,h_n,color="k",alpha=0.25)
		axes[1].plot(model.time[:-1],m_n,c="k",label="Estimated new exposures per day")
		axes[1].axvline(tr_start,c="k")
		axes[1].axvline(len(model.time)-model.D_e-1,c="k")

		## Some labels
		for i, ax in enumerate(axes):
			ax.legend(frameon=False,fontsize=28)
			ax.set_yticks([])
			ax.set_ylabel("Step {}".format(i+1),fontweight="bold")
		axes[-1].set_xlabel("Model time step")
		#axes[2].set(ylim=(model.S0*0.996,None))

		## Annotations for the vertical bars
		axis_to_data = axes[3].transAxes + axes[3].transData.inverted()
		axes[3].text(tr_start,axis_to_data.transform((0,0.5))[1],
					 "Analysis period begins\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=90)
		axes[3].text(len(model.time)-model.D_e-1,axis_to_data.transform((0,0.5))[1],
					 "Analysis period ends\n",fontsize=22,
					 horizontalalignment="center",verticalalignment="center",
					 rotation=-90)
		
		## Done
		fig.tight_layout()
		fig.savefig("..\\_plots\\debug.png")
		plt.show()
		sys.exit()

	return Y