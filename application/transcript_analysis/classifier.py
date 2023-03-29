import d6tflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


class LoadData(d6tflow.tasks.TaskPqPandas):
    def output(self):
        return d6tflow.targets.LocalTarget('data.pkl')

    def run(self):
        # Load the YouTube-8M dataset into memory
        # Preprocess the transcript data
        # Save the preprocessed data as a Pandas dataframe
        data = pd.read_csv('youtube-8m-dataset/annotations.csv')
        data = preprocess_data(data)
        self.save(data)


class GenerateFeatures(d6tflow.tasks.TaskPickle):
    def requires(self):
        return LoadData()

    def output(self):
        return d6tflow.targets.LocalTarget('features.pkl')

    def run(self):
        # Generate features from the preprocessed transcript data
        # Save the features as a sparse matrix
        data = self.input().load()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data['transcript'])
        self.save(X)


class SplitDataset(d6tflow.tasks.TaskPickle):
    def requires(self):
        return GenerateFeatures()

    def output(self):
        return {'train': d6tflow.targets.LocalTarget('train.pkl'),
                'val': d6tflow.targets.LocalTarget('val.pkl'),
                'test': d6tflow.targets.LocalTarget('test.pkl')}

    def run(self):
        # Split the dataset into training, validation, and testing sets
        # Save the split datasets as Pandas dataframes
        X = self.input().load()
        y = data['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.save({'train': (X_train, y_train),
                   'val': (X_val, y_val),
                   'test': (X_test, y_test)})


class TrainClassifier(d6tflow.tasks.TaskPickle):
    def requires(self):
        return SplitDataset()

    def output(self):
        return d6tflow.targets.LocalTarget('classifier.pkl')

    def run(self):
        # Train a classifier on the training set and tune hyperparameters on the validation set
        # Save the trained classifier as a serialized file
        X_train, y_train = self.input()['train'].load()
        X_val, y_val = self.input()['val'].load()

        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        self.save(clf)


class EvaluateClassifier(d6tflow.tasks.TaskCache):
    def requires(self):
        return {'classifier': TrainClassifier(),
                'test': SplitDataset()['test']}

    def run(self):
        # Evaluate the performance of the classifier on the testing set
        clf = self.input()['classifier'].load()
        X_test, y_test = self.input()['test'].load()

        score = clf.score(X_test, y_test)
        self.save(score)


class ReportResults(d6tflow.tasks.TaskPickle):
    def requires(self):
        return EvaluateClassifier()

    def output(self):
        return d6tflow.targets.LocalTarget('report.pkl')

    def run(self):
        # Generate a report on the performance of the classifier
        clf = self.input()['classifier'].load()
        X_test, y_test = self.input()['test'].load()

        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        with self.output().open('w') as f:
            f.write('Classification Report:\n')
            f.write(report)
            f.write('\n\nConfusion Matrix:\n')
            f.write(str(matrix))

