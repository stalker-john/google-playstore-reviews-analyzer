# Google Playstore Reviews Analyzer

---

## Using the Scraper

For using the scraper run the following command

```python
cd scraper
python scraper.py
```

The scraper will then collect data of multiple app belonging to different categories like finance, fashion, health etc and store them in their
respective csv files.

## Analzing the Dataset

For running the pipeline for analyzing the dataset, we need to run the commands to open jupyter notebook.
Ensure that you have jupyter notebook installed in your system.

```python
cd analysis
jupyter notebook
```

Following which jupyter notebook will open in the analysis directory. Open the analysis.ipynb file and run all the cells

## Web Application

For running the webapp successfully, you need to have installed Flask. You can check its installation [here](https://phoenixnap.com/kb/install-flask).

Run the following commands to host the app on localhost with default port 5000.

```python
cd web-app
python -m flask run
```

Then open your web browser and open localhost:5000

![Home](https://github.com/stalker-john/google-playstore-reviews-analyzer/blob/main/web-app/static/images/ss_home.png?raw=true)
![Output](https://github.com/stalker-john/google-playstore-reviews-analyzer/blob/main/web-app/static/images/ss_output.png?raw=true)
