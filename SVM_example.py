from SVM import SVM
from MultiClass import MC_Classifier
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

X,y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=22)

model = MC_Classifier(SVM)
model.fit(X_train, y_train, C=1000, kernel="rbf", gamma=0.001, solver_library="cvxopt", verbose=False)

y_pred = model.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred= y_pred))
print(f"Accuracy : {accuracy_score(y_true=y_test, y_pred=y_pred)}")