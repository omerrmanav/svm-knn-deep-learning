import pandas as pd
import numpy as np

def create_dataset():
    # np.random.seed(1) # aynı random sayıları üretmek için
    sample = 1000

    user_id = np.arange(0, sample)  
    age = np.random.randint(18, 70, size=sample)  
    purchase_frequency = np.random.randint(1, 20, size=sample)  # aylık alışveriş sıklığı
    total_spent = np.random.randint(30, 1500, size=sample)  # ylık harcama 

    high_spender = np.zeros(sample) # herkes düşük harcama olarak başlar 
    # aşağıdaki kontrolllerle güncellenir değeri
    high_spender[(age < 30) & (purchase_frequency > 10) & (total_spent > 500)] = 1
    high_spender[(age >= 60) & (purchase_frequency < 8) & (total_spent > 1000)] = 1
    high_spender[(age >= 30) & (age < 60) & (purchase_frequency > 12) & (total_spent > 800)] = 1

    data = {
        'User_ID': user_id,
        'Age': age,
        'Purchase_Frequency': purchase_frequency,
        'Total_Spent': total_spent,
        'High_Spender': high_spender  # Sınıflandırma hedefi: 1 = yüksek harcama, 0 = düşük harcama
    }

    df = pd.DataFrame(data)
    df.to_csv('C:\\Users\\Administrator\\Desktop\\Dosyalar\\Kodlar\\python\\data-mining\\sample_data.csv', index=False)
    print("Dataset succesfully saved.")

    return df

create_dataset()
