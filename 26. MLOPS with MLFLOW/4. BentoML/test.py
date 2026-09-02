import bentoml
iris_clf_runner = bentoml.sklearn.get("iris_clf:poaylangt6sjnybo").to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.4,3.,5.1,1.8]]))