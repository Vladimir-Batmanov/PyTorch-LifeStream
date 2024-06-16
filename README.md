
<img width="200" alt="image" src="https://github.com/Vladimir-Batmanov/PyTorch-LifeStream/assets/45069495/4fb16285-381c-438f-b650-ab0142174356">
<img width="270" alt="image" src="https://github.com/Vladimir-Batmanov/PyTorch-LifeStream/assets/45069495/a3c0fad6-bfa2-433a-a608-e4b4e0a398f2">



## Документация проекта

---

### Описание проекта

**Название проекта:** Предиктивная модель для рекомендации продуктов банка

**Цель проекта:**
Разработать мультимодальную модель, прогнозирующую покупку клиентом каждого из 4 продуктов при их предложении банком. 

### Содержание

1. Описание использованных методов обработки данных
2. Описание методов работы модели/ансамбля моделей
3. Оценка работы модели/ансамбля моделей с указанием итоговой точности модели
4. Скорость обучения и работы модели
5. Дополнительные условия и ограничения, введенные командой для решения задачи

---

### 1. Описание использованных методов обработки данных

#### Предварительная обработка данных

**1.1 Нормализация:**
- Нормализация данных была проведена для приведения всех признаков к единому масштабу. Это улучшает качество и стабильность обучения модели. Для этого использовалась стандартная нормализация, приводящая каждый признак к среднему значению 0 и стандартному отклонению 1.

**1.2 Агрегация и генерация признаков:**
- Агрегация данных включает создание новых признаков на основе имеющихся данных. Например, для каждого клиента рассчитывались суммы, средние значения, максимумы и минимумы транзакций за определенный период.
- Генерация признаков включала создание дополнительных метрик, таких как частота транзакций, общее количество транзакций и уникальные категории транзакций.

```python
import numpy as np
import pandas as pd

# Укажите путь к папке с файлами Parquet
folder_path = 'train_target_parquet'
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

# Объединение данных из всех файлов Parquet
combined_df = pd.DataFrame()
for file in parquet_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_parquet(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Пример агрегации данных
df_test = combined_df
df_test['period'] = pd.to_datetime(df_test['mon'])
df_test = df_test.drop(columns=['mon'])
```

#### 1.3 Разделение данных на части

- Данные были разделены на 25 частей для обработки. Каждая часть включала определенное количество клиентов для упрощения процесса обработки.

```python
# Разделение данных на части
unique_client_ids = df_test['client_id'].unique()
sample_size = len(unique_client_ids) // 30
dfs = []
for i in range(25):
    if i == 24:
        part_client_ids = unique_client_ids[i * sample_size:]
    else:
        part_client_ids = unique_client_ids[i * sample_size:(i + 1) * sample_size]
    part_df = df_test[df_test['client_id'].isin(part_client_ids)]
    dfs.append(part_df)
```

---

### 2. Описание методов работы модели/ансамбля моделей

#### Создание эмбеддингов с использованием PyTorch Lifestream

**2.1 Сбор последовательностей событий:**
- Для каждого клиента собирались данные о его действиях за период. Использовались такие события, как транзакции, диалоги с банком и гео-активность.

**2.2 Обучение модели для создания эмбеддингов:**
- Модель для создания эмбеддингов обучалась с использованием PyTorch и PyTorch Lightning. Использовалась библиотека PyTorch-LifeStream для обработки последовательностей событий и создания эмбеддингов.

```python
import torch
import pytorch

```python
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

class TransactionDataset(Dataset):
    def __init__(self, data, client_col='client_id', period_col='period', target_cols=None):
        self.data = data
        self.client_col = client_col
        self.period_col = period_col
        self.target_cols = target_cols if target_cols else []

        self.clients = data[client_col].unique()
        self.data = self.data.sort_values(by=[client_col, period_col])
        
    def __len__(self):
        return len(self.clients)
    
    def __getitem__(self, idx):
        client_id = self.clients[idx]
        client_data = self.data[self.data[self.client_col] == client_id]
        
        features = []
        for i in range(1, len(client_data) + 1):
            monthly_data = client_data.iloc[:i].drop(columns=[self.client_col, self.period_col] + self.target_cols)
            features.append(monthly_data.values)
        
        targets = client_data[self.target_cols].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), client_id

class EmbeddingModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(EmbeddingModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_packed = torch.nn.utils.rnn.pack_sequence([torch.tensor(f) for f in x], enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed)
        embeddings = self.fc(ht[-1])
        return embeddings

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        embeddings = self(x)
        loss = F.mse_loss(embeddings, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

**2.3 Формирование эмбеддингов:**
- Для каждого клиента создавались эмбеддинги на основе данных за год, начиная с информации только о первом месяце и постепенно добавляя данные о последующих месяцах.

```python
def train_model(data, input_size, hidden_size, output_size, num_layers=2, dropout=0.1, max_epochs=30):
    dataset = TransactionDataset(data, target_cols=['target_1', 'target_2', 'target_3', 'target_4'])
    
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = EmbeddingModel(input_size, hidden_size, output_size, num_layers, dropout)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else 1, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, valid_loader)
    
    return model, trainer

def generate_embeddings(model, data, input_size):
    dataset = TransactionDataset(data, target_cols=['target_1', 'target_2', 'target_3', 'target_4'])
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    embeddings = []
    client_ids = []

    for batch in data_loader:
        x, y, client_id = batch
        with torch.no_grad():
            embedding = model(x)
        embeddings.append(embedding.cpu().numpy())
        client_ids.extend(client_id.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return pd.DataFrame(embeddings, index=client_ids, columns=[f'emb_{i}' for i in range(input_size)])
```

#### Обучение модели CatBoost

**2.4 Разделение данных:**
- Данные были разделены на обучающую и тестовую выборки в пропорции 80/20.

**2.5 Обучение модели:**
- Модель CatBoost обучалась на обучающих данных с использованием эмбеддингов клиентов.

```python
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Создание и обучение модели
model = MultiOutputClassifier(CatBoostClassifier(iterations=100, learning_rate=0.1))
model.fit(X_train, y_train)

# Оценка производительности модели
roc_auc_scores = {}
accuracy_scores = {}
for i, target in enumerate(targets.columns):
    y_pred = model.predict(X_test)[:, i]
    accuracy = accuracy_score(y_test[target], y_pred)
    accuracy_scores[target] = accuracy
    print(f'Accuracy for {target}: {accuracy}')
    
    if len(y_test[target].unique()) == 2:
        y_pred_proba = model.estimators_[i].predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test[target], y_pred_proba)
        roc_auc_scores[target] = roc_auc
        print(f'ROC-AUC for {target}: {roc_auc}')
    else:
        print(f'ROC-AUC for {target} is not defined due to only one class present in y_test.')
        roc_auc_scores[target] = None
```

**2.6 Подбор гиперпараметров:**
- Использование библиотеки Optuna для оптимизации гиперпараметров.

```python
import optuna
from sklearn.metrics import log_loss

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
    }

    model = MultiOutputClassifier(CatBoostClassifier(**params, verbose=0))
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)
    log_losses = [log_loss(y_test.iloc[:, i], y_pred_proba[i]) for i in range(y_test.shape[1])]
    return sum(log_losses) / len(log_losses)

# Оптимизация гиперпараметров
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Лучшие параметры
best_params = study.best_params
print("Best parameters:", best_params)

# Обучение модели с лучшими параметрами на всех данных
best_model = MultiOutputClassifier(CatBoostClassifier(**best_params, verbose=0))
best_model.fit(features, targets)
```

---

### 3. Оценка работы модели/ансамбля моделей с указанием итоговой точности модели

**Метрики оценки:**
- Основная метрика: ROC-AUC (0.78)
- Вспомогательные метрики: Accuracy для каждого из 4 таргетов.

**Процесс оценки:**
- Оценка производительности модели на тестовой выборке с использованием ROC-AUC и Accuracy.
- Оценка производительности модели на новых данных и валидация результатов.

**Результаты:**
- ROC-AUC: 0.78
- Accuracy: Значения для каждого из 4 таргетов:
  - target_1: 0.9908534415260872
  - target_2: 0.9987543597409069
  - target_3: 0.9932023631575201
  - target_4: 0.9927041070538828

---

### 4. Скорость обучения и работы модели

**Скорость обучения:**
- Время обучения модели: 67 минут

**Скорость работы:**
- Время предсказания для одного клиента: меньше 30 секунд
- Время предсказания для всего тестового набора: 6 миннут

---

### 5. Дополнительные условия и ограничения, введенные командой для решения задачи

**Ограничения:**
- Данные были анонимизированы и деперсонализированы в соответствии с ФЗ 152 "О персональных данных".
- Обязательное использование библиотеки PyTorch-LifeStream для создания эмбеддингов.
- Модель обучалась только на предоставленных данных, без использования внешних источников информации.

**Дополнительные условия:**
- В случае использования дополнительных данных необходимо аргументировать и оценить их влияние на работу модели.
- Описание сборки и обработки данных для прозрачности и воспроизводимости решения.

---

### Заключение

Разработанная модель демонстрирует высокую точность в прогнозировании покупок клиентов. Использование мультимодальных данных и современных методов машинного обучения позволило достичь значительного улучшения качества предсказаний. В дальнейшем планируется валидация модели на новых данных и внедрение в производственную среду.

---

### Контакты

- Telegram: Алеев Андрей Романович - [@andrewaleev](https://t.me/andrewaleev)
- Telegram: Батманов Владимир Андреевич - [@vbatmanov](https://t.me/vbatmanov)
- Telegram: Селищев Евгений Викторович - [@e_selishchevx](https://t.me/e_selishchevx)
- Telegram: Сушенцев Алексей Артемович - [@sush_385](https://t.me/sush_385)
- Telegram: Салаев Тимур Эльдарович - [@monster2882](https://t.me/monster2882)
