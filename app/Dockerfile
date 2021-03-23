FROM python:3.7
EXPOSE 8501
WORKDIR .
COPY . .
RUN pip3 install -r requirements.txt
ENTRYPOINT streamlit run app.py
