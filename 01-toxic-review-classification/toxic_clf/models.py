from pathlib import Path
from statistics import mean, stdev

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def experiments_lr(dataset):
    X = dataset['message']
    y = dataset['is_toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=10, shuffle=True)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    param_grid = {
        'tfidf__max_features': [1000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['liblinear', 'lbfgs']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

def experiments_rf(dataset):
    X = dataset['message']
    y = dataset['is_toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=10, shuffle=True)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])

    param_grid = {
        'tfidf__max_features': [1000, 5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


def classifier(dataset, model):
    X = dataset['message']
    y = dataset['is_toxic']

    if model == 'classic_ml':
        # Преобразование текста в числовое представление

        vectorizers = {
            'tfidf': TfidfVectorizer(),
            'count': CountVectorizer()
        }

        models = {
            'rf': RandomForestClassifier(),
            'lr': LogisticRegression()
        }

        for vect_name, vectorizer in vectorizers.items():
          X_vec = vectorizer.fit_transform(X)
          for model_name, model_instance in models.items():
              X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
              model_instance.fit(X_train, y_train)

              kf = KFold(n_splits=10, shuffle=True)
              scores = cross_val_score(model_instance, X_train, y_train, cv=kf, scoring='f1')
              print(f"{vect_name} - {model_name} - f1: {scores.mean():.4f}")


              y_pred = model_instance.predict(X_test)
              print(confusion_matrix(y_test, y_pred))

    elif model == 'microsoft/codebert-base':
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx]).to(device)
                return item

            def __len__(self):
                return len(self.labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')
        test_encodings = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')

        train_dataset = Dataset(train_encodings, y_train)
        test_dataset = Dataset(test_encodings, y_test)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            dataloader_pin_memory=False,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        codebert_eval_results = trainer.evaluate()
        print(codebert_eval_results)
    else:
        raise ValueError("Invalid model type")

