import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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

#### Gaussian noise ####
def add_noise(x, y, intervals, x_var, y_var):
    perturbed_x = np.array([xi + np.random.normal(loc=0, scale=np.sqrt(x_var[j]))
                            for xi in x for j, (lower, upper) in enumerate(intervals) if lower <= xi < upper])
    perturbed_y = np.array([yi + np.random.normal(loc=0, scale=np.sqrt(y_var[j]))
                            for yi, xi in zip(y, x) for j, (lower, upper) in enumerate(intervals) if lower <= xi < upper])
    return perturbed_x, perturbed_y

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
sample_size = 10000
x= dfu['BodyweightKg']
y= dfu['TotalKg']
x.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
sample_indices = np.random.choice(len(x), size=sample_size, p=weights)
x_sampled = x[sample_indices]
y_sampled = y[sample_indices]

# Noise (temporary setup)
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
plt.plot(x_grid,kde_values_grid, label='KDE') 
bins=np.linspace(min(dfu['BodyweightKg']), max(dfu['BodyweightKg']), 100)
plt.hist(dfu['BodyweightKg'], bins=bins , density=True, edgecolor=(0, 0, 0, 1), facecolor=(1, 1, 1, 1), label='BW Distribution')
plt.legend()
plt.show()
plt.rcParams.update({'font.size': 20})
plt.plot(x_sampled, y_sampled, marker='+', linestyle='None', markersize=10, markeredgewidth=1, color='black')
plt.xlabel('Bodyweight (kg)')
plt.ylabel('Total (kg)')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt), label='Logistic') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt2), label='GL')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt3), label='Logistic (resampling)') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt4), label='GL (resampling)')
plt.xlim([0, 250])
plt.gcf().set_size_inches(15, 6)
plt.legend()
plt.show()

#### Plot fit and original dataset ####
plt.rcParams.update({'font.size': 20})
plt.plot(dfu['BodyweightKg'], dfu['TotalKg'], marker='+', linestyle='None', markersize=10, markeredgewidth=1, color='black')
plt.xlabel('Bodyweight (kg)')
plt.ylabel('Total (kg)')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt), label='Logistic') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt2), label='GL')
plt.plot(list(range(0, 251, 1)), logistic(list(range(0, 251, 1)), *popt3), label='Logistic (resampling)') 
plt.plot(range(0,251,1), GL(range(0,251,1),*popt4), label='GL (resampling)')
plt.xlim([0, 250])
plt.gcf().set_size_inches(15, 6)
plt.legend()
plt.show()

#### Analysis of the bodyweight distribution in the resampled dataset (it should be roughly uniform) ####
plt.hist(x_sampled, bins=100, color='white', edgecolor='black', density=True)
plt.xlabel('Bodyweight (Kg)')
plt.ylabel('Frequency')
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
plt.gcf().set_size_inches(15, 6)
plt.title('Distribution of score')
plt.show()

#### Analysis of the score distribution in the IPF weight classes (original dataset) (the boxplots should be aligned if the score is unbiased) ####
if user_input == 'F':
    classes = [0,47,52,57,63,69,76,84,np.max(dfu['BodyweightKg'])]
else :
    classes = [0,59,66,74,83,93,105,120,np.max(dfu['BodyweightKg'])]
dfu['class'] = pd.cut(dfu['BodyweightKg'], bins=classes, right=False, include_lowest=True)
dfu.boxplot(column='score', by='class', grid=False)
plt.title('Distribution of score in the IPF classes (original dataset)')
plt.show()
