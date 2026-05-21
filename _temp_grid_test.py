import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier


df = pd.read_csv('./data/titanic_procesado.csv')
X = df.drop(['Survived'], axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
modelos = {
    'Clasificador XGBoost': {
        'modelo': XGBClassifier(),
        'parametros': {
            'n_estimators': [10, 100],
            'max_depth': [None, 1, 2, 3]
        }
    }
}

for nombre, info_modelo in modelos.items():
    print('test', nombre)
    grid_search = GridSearchCV(estimator=info_modelo['modelo'], param_grid=info_modelo['parametros'], cv=2, scoring='accuracy', n_jobs=-1)
    try:
        grid_search.fit(X_train, y_train)
        print('ok')
    except Exception as e:
        import traceback; traceback.print_exc()
