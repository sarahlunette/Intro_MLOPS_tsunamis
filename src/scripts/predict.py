
def predict_record(record, model):
	pred = model.predict(record)
	return pred