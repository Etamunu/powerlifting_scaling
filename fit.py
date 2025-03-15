import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.stats import norm

#### Data loading ####
file_path = 'openpower-filtered.csv' # Path to your CSV file
df = pd.read_csv(file_path)
while True:
    user_input = input("Please enter 'F' for female or 'M' for male: ").upper()
    if user_input == 'F':
        dfu = df[df['Sex'] == 'F']
        break
    elif user_input == 'M':
        dfu = df[df['Sex'] == 'M']
        break
    else:
        print("Invalid input. Please enter 'M' or 'F'.")
 
#### Functions to fit ####
def logistic(x, L, k, x0):
    return (L/ (1 + np.exp(-k * (x - x0))))-(L/ (1 + np.exp(k * x0)))

def GL(x, L, x0, k): # Mimics the function used for the IPF GL score.
    return L-np.exp(-k* x+x0)

def add_noise(x, y, intervals, x_var, y_var, correlation=0.5): # Gaussian noise to improve the visual after resampling
    perturbed_x = []
    perturbed_y = []
    for xi, yi in zip(x, y):
        for j, (lower, upper) in enumerate(intervals):
            if lower <= xi < upper:
                cov_matrix = [[x_var[j], correlation * np.sqrt(x_var[j] * y_var[j])],
                              [correlation * np.sqrt(x_var[j] * y_var[j]), y_var[j]]]
                noise_x, noise_y = np.random.multivariate_normal(mean=[0, 0], cov=cov_matrix)
                perturbed_x.append(xi + noise_x)
                perturbed_y.append(yi + noise_y)
    return np.array(perturbed_x), np.array(perturbed_y)

#### Fit the original dataset ####
popt, pcov = curve_fit(logistic, dfu['BodyweightKg'], dfu['TotalKg'], p0=[np.max(dfu['TotalKg']) , 0.01, np.mean(dfu['BodyweightKg'])])
popt2, pcov2 = curve_fit(GL, dfu['BodyweightKg'], dfu['TotalKg'], p0=[0. , 0., 0.])

#### Resampling ####
kde = gaussian_kde(dfu['BodyweightKg'])
x_grid = np.linspace(min(dfu['BodyweightKg']), max(dfu['BodyweightKg']), 1000)
kde_values_grid = kde(x_grid)
interp_kde = interp1d(x_grid, kde_values_grid)
kde_values = interp_kde(dfu['BodyweightKg'])
weights = 1. / kde_values
weights = weights / np.sum(weights)
sample_size = 100000
x= dfu['BodyweightKg']
y= dfu['TotalKg']
x.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
sample_indices = np.random.choice(len(x), size=sample_size, p=weights)
x_sampled = x[sample_indices]
y_sampled = y[sample_indices]

#### Noise (to help in the visualisation) ####
noise_var = {}
classes_limits= [0,50,75,np.max(dfu['BodyweightKg'])]
classes = [(classes_limits[i], classes_limits[i+1]) for i in range(len(classes_limits) - 1)]
x_var = []
y_var = []
for lower, upper in classes:
    x_c = x_sampled[(x_sampled >= lower) & (x_sampled < upper)]
    y_c = y_sampled[(x_sampled >= lower) & (x_sampled < upper)]
    if len(x_c) > 1:
        x_v = np.sqrt(np.min(x_c))
        y_v = np.sqrt(np.min(y_c))
    else:
        variance = 0
    x_var.append(x_v)
    y_var.append(y_v)

x_sampled, y_sampled = add_noise(x_sampled, y_sampled, classes, x_var, y_var)

#### Fit the resampled dataset ####
popt3, pcov3 = curve_fit(logistic, x_sampled, y_sampled, p0=[np.max(y) , 0.01, np.mean(x)])
popt4, pcov4 = curve_fit(GL, x_sampled, y_sampled, p0=[0. , 0., 0.])

#### Plot fit and resampled dataset ####
plt.xlabel("Body Weight (kg)")
plt.ylabel("Frequency")
plt.plot(x_grid,kde_values_grid, label='KDE')
bins=np.linspace(min(dfu['BodyweightKg']), max(dfu['BodyweightKg']), 100)
plt.hist(dfu['BodyweightKg'], bins=bins , density=True, edgecolor=(0, 0, 0, 1), facecolor=(1, 1, 1, 1), label='BW Distribution')
plt.legend()
plt.title('Distribution of Bodyweight')
plt.show()
plt.plot(x_sampled, y_sampled, marker='o', linestyle='None', markersize=1, markeredgewidth=1, color='black', alpha = 0.4)
plt.xlabel('Bodyweight (kg)')
plt.ylabel('Total (kg)')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt), label='Logistic', linewidth=2) 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt2), label='GL', linewidth=2)
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt3), label='Logistic (resampling)', linewidth=2) 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt4), label='GL (resampling)', linewidth=2)
plt.xlim([0, 250])
plt.legend()
plt.title('Total vs Bodyweight (resampled dataset)')
plt.show()

#### Plot fit and original dataset ####
plt.plot(dfu['BodyweightKg'], dfu['TotalKg'], marker='o', linestyle='None', markersize=1, markeredgewidth=1, color='black', alpha = 0.4)
plt.xlabel('Bodyweight (kg)')
plt.ylabel('Total (kg)')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt), label='Logistic') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt2), label='GL')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt3), label='Logistic (resampling)') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt4), label='GL (resampling)')
plt.xlim([0, 250])
plt.legend()
plt.title('Total vs Bodyweight (original dataset)')
plt.show()

#### Analysis of the bodyweight distribution in the resampled dataset (it should be roughly uniform) ####
plt.hist(x_sampled, bins=100, color='white', edgecolor='black', density=True)
plt.title('Distribution of Bodyweight (resampled dataset)')
plt.gcf().set_size_inches(15, 6)
plt.show()

#### Analysis of the score distribution (it should be centered on 1) ####
dfu['score'] = dfu['TotalKg']/logistic(dfu['BodyweightKg'], *popt3)
dfu=dfu[~np.isnan(dfu['score'])]  
plt.hist(dfu['score'], bins=100, color='white', edgecolor='black', density=True)
mu, std = norm.fit(dfu['score'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

# Examples of athletes
if user_input == 'F':
    Chapon = 431.5/logistic(46.77, *popt3)
    plt.axvline(x=Chapon, color='red')
    plt.text(Chapon+0.01, 1, s='Tiffany Chapon', rotation=90, verticalalignment='center')
    Schlater = 710/logistic(126.56, *popt3)
    plt.axvline(x=Schlater, color='red')
    plt.text(Schlater+0.01, 1, s='Brittany Schlater', rotation=90, verticalalignment='center')
    Tongotea = 610.5/logistic(75.6, *popt3)
    plt.axvline(x=Tongotea, color='red')
    plt.text(Tongotea+0.01, 1, s='Karlina Tongotea', rotation=90, verticalalignment='center')
else:
    Olivares = 1152.5/logistic(178.2, *popt3)
    plt.axvline(x=Olivares, color='red')
    plt.text(Olivares+0.01, 1, s='Jesus Olivares', rotation=90, verticalalignment='center')
    Tarinidis = 707.5/logistic(65.95, *popt3)
    plt.axvline(x=Tarinidis, color='red')
    plt.text(Tarinidis+0.01, 1, s='Panagiotis Tarinidis', rotation=90, verticalalignment='center')
    Fedosienko = 669.5/logistic(58.48, *popt3)
    plt.axvline(x=Fedosienko, color='red')
    plt.text(Fedosienko+0.01, 1, s='Sergei Fedosienko', rotation=90, verticalalignment='center')
    
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Distribution of score')
plt.show()

#### Analysis of the score distribution in the IPF weight classes (original dataset) (the rolling quantiles should be aligned if the score is unbiased) ####
dfu = dfu.sort_values('BodyweightKg')
window_size = 100

# Calculate rolling median and quantiles (5th, 10th, 25th, 50th, 75th, 90th, and 95th percentiles)
dfu['median'] = dfu['score'].rolling(window=window_size, center=True).median()
dfu['q5'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.05)
dfu['q10'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.10)
dfu['q25'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.25)
dfu['q75'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.75)
dfu['q90'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.90)
dfu['q95'] = dfu['score'].rolling(window=window_size, center=True).quantile(0.95)

# Plot the rolling statistics
plt.figure(figsize=(10, 6))

# Color scheme
darker_blue = '#1f77b4'
darker_fill_10_90 = '#1f77b4'
darker_fill_25_75 = '#1f77b4'

plt.plot(dfu['BodyweightKg'], dfu['median'], label='Median', color=darker_blue)
plt.fill_between(dfu['BodyweightKg'], dfu['q25'], dfu['q75'], color=darker_fill_25_75, alpha=0.4)
plt.fill_between(dfu['BodyweightKg'], dfu['q10'], dfu['q90'], color=darker_fill_10_90, alpha=0.2)
plt.fill_between(dfu['BodyweightKg'], dfu['q5'], dfu['q95'], color=darker_fill_10_90, alpha=0.1)

plt.xlabel('Body Weight (Kg)', fontsize=30)
plt.ylabel('Score', fontsize=30)
plt.title('Rolling Quantiles', fontsize=30)

loc_x = np.min(dfu['BodyweightKg'])+15

plt.text(loc_x, dfu['median'].median(), '50%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q25'].median(), '25%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q75'].median(), '75%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q10'].median(), '10%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q90'].median(), '90%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q5'].median(), '5%', color='blue', verticalalignment='center', fontsize=22)
plt.text(loc_x, dfu['q95'].median(), '95%', color='blue', verticalalignment='center', fontsize=22)

# Show the plot
plt.show()
