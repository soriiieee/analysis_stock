import os, sys
import time
import shutil

from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
options = Options()

import subprocess


class Octo:
  def __init__(self,download_dir):
    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2)
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir',download_dir)
    profile.set_preference("browser.helperApps.alwaysAsk.force", False)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', ('application/octet?stream;charset=UTF-8'))
    self.driver = webdriver.Firefox(firefox_profile=profile, executable_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\geckodriver.exe")
    self.driver.implicitly_wait(10)
  
  def get_csv(self,category_id,area_id):
    self.driver.get("http://occtonet.occto.or.jp/public/dfw/RP11/OCCTO/SD/LOGIN_login#")
    WebDriverWait(self.driver, 5).until(lambda d: len(d.window_handles) > 1)
    handles = self.driver.window_handles
    
    self.driver.switch_to.window(handles[0])
    self.driver.find_element_by_id("menu1-7").click()
    self.driver.find_element_by_id("menu1-7-1-1").click()
    time.sleep(10)

    handles = self.driver.window_handles
    self.driver.switch_to.window(handles[2])
    print(self.driver.current_url) #debug.....
    try:
      self.getInfoBySelenium(self.driver, category_id=category_id, area_id=area_id)
      self.driver.quit()
    except:
      time.sleep(2)
      self.getInfoBySelenium(self.driver, category_id=category_id, area_id=area_id)
      self.driver.quit()
    return

  def getInfoBySelenium(self,driver,category_id="rklDataKnd5",area_id="rkl2"):
    # driver.switch_to.window(handles[1]) #latest browser
    # wait = WebDriverWait(driver, 5)
    # el = wait.until(EC.element_to_be_clickable((By.ID, 'rklDataKnd1')))
    """
    update!!  2021.10.06
    """
    #---
    #category id 
    # "rklDataKnd2" : 月間情報
    # "rklDataKnd3" : 週間情報
    # "rklDataKnd4" : 翌々日情報
    # "rklDataKnd5" : 翌日情報
    #---
    self.driver.find_element_by_id(category_id).click()
    #all year... too heavy datas...
    self.driver.find_element_by_id("rklNngpFrom").click()#calender..
    handles2 = self.driver.window_handles
    # driver.switch_to.window(handles2[1])
    
    # class="ui-datepicker-year"
    self.driver.find_element_by_id("ui-datepicker-div").click()
    
    # self.driver.find_element_by_link_text("1").click()
    sys.exit()
    # driver.switch_to.window(handles2[0])
    # driver.find_element_by_id("rklAllTermDwld").click()
    self.driver.find_element_by_id(area_id).click()
    self.driver.find_element_by_id("csvBtn").click()
    # Alert(driver).accept()
    time.sleep(1)

    ele = self.driver.find_element_by_class_name('ui-dialog-buttonset')
    _btn = ele.find_elements_by_tag_name("button")
    # print(len(_btn))
    _btn[0].click()
    time.sleep(3)
    return


def set_param(f,a):
  FCT={
    "1M" :"rklDataKnd2", # "rklDataKnd2" : 月間情報
    "1W" :"rklDataKnd3", # "rklDataKnd3" : 週間情報
    "2D" :"rklDataKnd4", # "rklDataKnd4" : 翌々日情報
    "1D" :"rklDataKnd5", # "rklDataKnd5" : 翌日情報
  }
  AREA ={
    "1":"rkl1", #北海道-本州間
    "2":"rkl2", #東北-東京間
    "3":"rkl3", #東京-中部間
    "4":"rkl4", #中部-関西間
    "5":"rkl5", #中部-北陸間
    "6":"rkl6", #北陸-関西間
    "7":"rkl7", #関西-中国間
    "8":"rkl8", #関西-四国間
    "9":"rkl9", #中国-四国間
    "10":"rkl10", #中国-九州間
    "11":"rkl11" #中部・関西-北陸間
  }
  cate = FCT[f]
  area = AREA[a]
  return cate,area



DOWNLOAD_DIR = os.getcwd() + "\\dat"
OUT_DIR = os.getcwd() + "\\out"
def main():
  """
  2020.11.03 init make class
  2021.10.08 update !
  """
  f,a = "1D","10"
  cate,area = set_param(f,a)
  oct1 = Octo(download_dir=DOWNLOAD_DIR)
  oct1.get_csv(category_id=cate, area_id=area)

  #rename and store data....
  rename_path = f".\\out\\{cate}_{area}.csv"
  
  sys.exit()
  
  
  load_data = f"{download_dir}\\{os.listdir(download_dir)[0]}"
  os.rename(load_data, rename_path)
  #debug....
  now = datetime.now()
  print(f"{now}[info]: end {category_id} - {area_id}")
  sys.exit()
  return



if __name__ == "__main__":
  #loop setting...
  #initial setting...
  # driver_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\geckodriver.exe"
  if 0: #cleaning init
    shutil.rmtree(DOWNLOAD_DIR)
    shutil.rmtree(OUT_DIR)
    os.mkdir(DOWNLOAD_DIR)
    os.mkdir(OUT_DIR)
  # subprocess.run("rm -f *.csv", cwd=download_dir, shell=True)
  # subprocess.run("rm -f *.csv", cwd=out_dir, shell=True)

  if 1:
    main()




