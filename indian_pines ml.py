import scipy.io
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
gt=scipy.io.loadmat("Indian_pines_gt")
corr=scipy.io.loadmat("Indian_pines_corrected")

y=gt["indian_pines_gt"]
X=corr["indian_pines_corrected"]

plt.plot([i for i in range(0,200)],corrected_pines[int(random.choices([i for i in range(0,145)])[0]),int(random.choices([i for i in range(0,145)])[0]),:])

plt.show()

X=np.concatenate(([X[:,i,:] for i in range(0,145)]),axis=0)
y=np.concatenate(([y[:,i] for i in range(0,145)]),axis=0)
X.shape
y.shape

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)
X_train, X_test, y_train , y_test = train_test_split(X_resampled, y_resampled, test_size=0.33, random_state=42)


classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
model_evaluation(y_predict,y_test)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_predict),display_labels=classifier.classes_)
disp.plot()
plt.show()


#print(sorted(Counter(y_resampled).items()))






