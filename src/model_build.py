from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier

def model_build(X_train,X_test,y_train,y_test):

    model = RandomForestClassifier(random_state=1)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test) 

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)

    print("Accuracy:",accuracy)
    print("Classification Report:\n",report)

    return model, accuracy