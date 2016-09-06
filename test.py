from selenium import webdriver
import time

browser = webdriver.Firefox()
browser.get('https://mail.google.com/')
emailElem = browser.find_element_by_id('Email')
emailElem.send_keys('f2015773@pilani.bits-pilani.ac.in')
emailElem.submit()
time.sleep(3)
passwordElem = browser.find_element_by_id('Passwd')
passwordElem.send_keys('like i will tell')
passwordElem.submit()
time.sleep(45)
compose = browser.find_element_by_class_name('T-I J-J5-Ji T-I-KE L3')
compose.click()
