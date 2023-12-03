from faker import Faker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

fake = Faker()

#setul de date va avea structura din cerinta, prin
#urmare trebuie creat

data = {'Neighborhood': ['A', 'B', 'C', 'D', 'E'],
        'Price': [100, 120, 80, 150, 90],
        'Occupancy': [80, 60, 20, 50, 10],
        'Review_Score': [4.5, 4.2, 4.8, 3.9, 4.6]}

for _ in range(150):
    #se pp ca exista doar aceste 5 zone: A, B, C, D, E
    data['Neighborhood'].append(random.choice(['A', 'B', 'C', 'D', 'E']))
    data['Price'].append(random.randint(1, 500))
    data['Occupancy'].append(random.randint(1, 100))
    data['Review_Score'].append(round(random.uniform(0, 5), 2))

df = pd.DataFrame(data)


plt.figure(figsize=(12, 6))
df_per_zones = df.groupby(['Neighborhood'])['Price'].mean().reset_index()
sns.barplot(df_per_zones, x = 'Neighborhood', y = 'Price')
plt.title('Pret pe zone la chirii')
plt.xlabel('Neighborhood')
plt.ylabel('Price')
#plt.show()

plt.figure(figsize=(12, 6))
df_per_zones = df.groupby(['Neighborhood'])['Review_Score'].mean().reset_index()
sns.barplot(df_per_zones, x = 'Neighborhood', y = 'Review_Score')
plt.title('Review pe zone la chirii')
plt.xlabel('Neighborhood')
plt.ylabel('Review_Score')
#plt.show()

#teoretic cea mai buna zona ar fi cea care ar avea media review urilor cea mai mare
#si media preturilor cea mai mica, adica maximizarea raportului dintre pret si review

price = df.groupby('Neighborhood')['Price'].mean()
review_score = df.groupby('Neighborhood')['Review_Score'].mean()

raport_review_price = review_score / price

plt.figure(figsize=(10, 6))
plt.bar(raport_review_price.index, raport_review_price.values)
plt.title('Raport calitate-pret pe zone')
plt.xlabel('Neighborhood')
plt.ylabel('Review_Score / Price')
#plt.show()

for id in ['A', 'B', 'C', 'D', 'E']:

    data = df[df['Neighborhood']==id]

    plt.figure(figsize=(12, 6))
    mean_per_occupancy = data.groupby('Occupancy')['Price'].mean().reset_index()
    sns.barplot(mean_per_occupancy, x = 'Occupancy', y = 'Price')
    plt.title('Venituri in functie de capacitate')
    plt.xlabel('Occupancy')
    plt.ylabel('Price')
    # plt.show()

    mean_per_review = data.groupby('Occupancy')['Review_Score'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(mean_per_review, x='Occupancy', y='Review_Score')
    plt.title('Review in functie de capacitate')
    plt.xlabel('Occupancy')
    plt.ylabel('Review')
    # plt.show()
