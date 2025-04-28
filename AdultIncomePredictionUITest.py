import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

logging.getLogger('selenium').setLevel(logging.WARNING)

def test_ui():
    chrome_options = Options()
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("http://127.0.0.1:5000")

    # Fill the form with test data
    driver.find_element(By.NAME, "age").send_keys("39")
    driver.find_element(By.NAME, "workclass").send_keys("Private")
    driver.find_element(By.NAME, "fnlwgt").send_keys("234721")
    driver.find_element(By.NAME, "education").send_keys("Bachelors")
    driver.find_element(By.NAME, "education.num").send_keys("13")
    driver.find_element(By.NAME, "marital.status").send_keys("Married")
    driver.find_element(By.NAME, "occupation").send_keys("Tech-support")
    driver.find_element(By.NAME, "relationship").send_keys("Husband")
    driver.find_element(By.NAME, "race").send_keys("White")
    driver.find_element(By.NAME, "sex").send_keys("Male")
    driver.find_element(By.NAME, "native.country").send_keys("United-States")

    # Submit the form
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

    # Wait for the prediction result to be visible
    try:
        wait = WebDriverWait(driver, 20)
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".prediction-results h3")))
        prediction_text = driver.find_element(By.CSS_SELECTOR, ".prediction-results h3").text.strip()
        print("Prediction:", prediction_text)
    except Exception as e:
        print("Error: Prediction element was not visible in time or not found.")
        print(e)

    # Optional: Close the browser after a delay to see the results
    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
    test_ui()