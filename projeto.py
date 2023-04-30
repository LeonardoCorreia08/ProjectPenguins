# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 19:02:19 2023

@author: My
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

pinguins = pd.read_csv("penguins.csv", sep=",")
df = pinguins.copy()

# Tratamento de valores faltantes
df['bill_length_mm'] = df['bill_length_mm'].fillna((df['bill_length_mm'].shift() + df['bill_length_mm'].shift(-1))/2)
df['bill_depth_mm'] = df['bill_depth_mm'].fillna((df['bill_depth_mm'].shift() + df['bill_depth_mm'].shift(-1))/2)
df['flipper_length_mm'] = df['flipper_length_mm'].fillna((df['flipper_length_mm'].shift() + df['flipper_length_mm'].shift(-1))/2)
df['body_mass_g'] = df['body_mass_g'].fillna((df['body_mass_g'].shift() + df['body_mass_g'].shift(-1))/2)
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

# Conversão de variáveis categóricas
df["species"] = df["species"].replace({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
df["island"] = df["island"].replace({"Biscoe": 0, "Dream": 1, "Torgersen": 2})
df["sex"] = df["sex"].str.lower().replace({"male": 0, "female": 1}).astype(int)

# Criação da árvore de decisão
X = pd.get_dummies(df.drop("species", axis=1))
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Seleção de amostras com filtros da árvore de decisão
selected_samples = X_test[(X_test['bill_depth_mm'] > 16.0) & (X_test['flipper_length_mm'] > 190)]

# Plot dos dados selecionados
plt.figure(figsize=(8,6))
plt.scatter(selected_samples['bill_length_mm'], selected_samples['bill_depth_mm'], c=y_test[selected_samples.index])
plt.xlabel('Comprimento do bico (mm)')
plt.ylabel('Profundidade do bico (mm)')
plt.title('Amostras Selecionadas de Pinguins')
plt.show()

# Visualização da matriz de correlação
correlation = df.corr()
plot = sns.heatmap(correlation, annot = True, fmt=".1f", linewidths=0.6)
plot

# Plot de diferentes gráficos
sns.relplot(x="flipper_length_mm", y="body_mass_g", data=pinguins, hue="island", size="bill_length_mm", palette='muted', height=7);

g = sns.jointplot(data=pinguins, x="body_mass_g", y="flipper_length_mm", hue="species", kind='kde');

sns.pairplot(pinguins, hue="island");

fig = plt.figure(figsize=(12, 8))
ax = plt.axes()

sns.set_theme(style="ticks", palette="pastel")

sns.boxplot(x="species", y="body_mass_g", data=pinguins, ax=ax)
sns.despine(offset=10, trim=True)

plt.show()

#Gráfico de dispersão com relação ao comprimento da nadadeira e à massa corporal, diferenciando por ilha e tamanho do bico
sns.relplot(x="flipper_length_mm", y="body_mass_g", data=pinguins, hue="island", size="bill_length_mm", palette='muted', height=7);

#Gráfico de densidade conjunta com relação à massa corporal e ao comprimento da nadadeira, diferenciando por espécie
g = sns.jointplot(data=pinguins, x="body_mass_g", y="flipper_length_mm", hue="species", kind='kde');

#Gráfico de pares com relação a todas as variáveis, diferenciando por ilha
sns.pairplot(pinguins, hue="island");

#Gráfico de caixa com relação à massa corporal e à espécie
fig = plt.figure(figsize=(12, 8))
ax = plt.axes()

sns.set_theme(style="ticks", palette="pastel")

sns.boxplot(x="species", y="body_mass_g", data=pinguins, ax=ax)
sns.despine(offset=10, trim=True)

plt.show()








