import uvicorn
from fastapi import FastAPI
import joblib,os

app = FastAPI()

#pkl
phish_model = open('phishing_k.pkl','rb')
phish_model_ls = joblib.load(phish_model)

# ML Aspect
@app.get('/predict/{site}')
async def predict(URL):
	X_predict = []
	X_predict.append(str(URL))
	y_Predict = phish_model_ls.predict(X_predict)
	if y_Predict == 'bad':
		result = "This is a Phishing Site"
	else:
		result = "This is not a Phishing Site"

	return (URL, result)
if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)