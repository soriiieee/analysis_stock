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

"""
参照記事
〇firefoxで自動ダウンロード(確認画面を消す)
  https://qiita.com/youngsend/items/25ac8ea6c176182db8f4
〇複数タブで同時に開くときにポップアップをブロックされる問題を解決する
  https://www.soudegesu.com/post/python/selenium-firefox-tab-restriction/
〇子を要素として、親の要素をを見つける
  https://qiita.com/ShortArrow/items/ced7bf23b806c0d835da
〇複数クラスのclassに関してを抽出する記述
https://qiita.com/hanonaibaobabu/items/e547410865d857aa25ec
"""

class Sfc:
  def __init__(self,datadir):
    #firefox setting
    fp = webdriver.FirefoxProfile()
    fp.set_preference('browser.download.folderList', 2) #0:desctop/1:システム規定フォルダ/
    fp.set_preference('browser.download.dir',datadir)
    # fp.set_preference('browser.helperApps.neverAsk.saveToDisk',('application/octet?stream;charset=UTF-8'))
    """
    sorimachiteの手順
    https://qiita.com/youngsend/items/25ac8ea6c176182db8f4 に記載
    firefox開発環境のネットワーク部分を参照する。実際のエクセルファイルの形式をcheckする。
    content-type は複数あるので、application/の方に注目する
    """
    #firefox ブラウザで読み込んだ時に、popup表示しないように対象ファイルの型を登録しておく
    # fp.set_preference('browser.helperApps.neverAsk.saveToDisk','text/x-comma-separated-values')
    fp.set_preference('browser.helperApps.neverAsk.saveToDisk',"application/octet-stream,text/csv,text/x-comma-separated-values,text/plain,application/x-msdownload,application/binary, text/csv, application/csv, application/excel, text/comma-separated-values, text/xml, application/xml,application/x-www-form-urlencoded,attachment/csv,attachment/x-comma-separated-values,application/x-comma-separated-values")
    fp.set_preference('browser.download.manager.showWhenStarting', False)
    fp.set_preference("browser.download.manager.alertOnEXEOpen", False)
    # fp.set_preference("browser.download.manager.showWhenStarting", False)
    fp.set_preference("browser.helperApps.alwaysAsk.force", False)
    driver_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\geckodriver.exe"
    
    self.driver = webdriver.Firefox(firefox_profile=fp, executable_path=driver_path,options=options)
    self.driver.implicitly_wait(10)
    
    # self.month = month
    # self.ff = ff
  def get(self,month,ff):
    # self.get_scode(ff)
    self.driver.get("https://www.data.jma.go.jp/gmd/risk/obsdl/index.php")
    
    # WebDriverWait(self.driver, 5).until(lambda d: len(d.window_handles) > 1)
    # handles = self.driver.window_handles
    # sys.exit()
    # self.driver.switch_to.window(handles[0])
    """地点検索"""
    if 1:
      self.driver.find_element_by_id(f"pr{ff}").click()
      # v = self.driver.find_element_by_id("stationArea").find_element_by_name("stid")
      element_text = "//div[input[@name='stid'][@value='s{}']]".format(47401)
      e = self.driver.find_element_by_xpath(element_text)
      e.click()
      time.sleep(1)
    
    """要素選定"""
    if 1:
      self.driver.find_element_by_id(f"elementButton").click()
      #1jikann 毎
      self.driver.find_element_by_xpath("//input[@name='aggrgPeriod'][@value='9'][@type='radio']").click()
      #要素選定
      self.driver.find_element_by_xpath("//input[@id='気温'][@value='201']").click()
      self.driver.find_element_by_xpath("//input[@id='全天日射量'][@value='610']").click()
      self.driver.find_element_by_xpath("//input[@id='風向・風速'][@value='301']").click()
      self.driver.find_element_by_xpath("//input[@id='現地気圧'][@value='601']").click()
      self.driver.find_element_by_xpath("//input[@id='降水量'][@value='101']").click()
      self.driver.find_element_by_xpath("//input[@id='相対湿度'][@value='605']").click()
      time.sleep(1)
      
    """期間選定"""
    if 1:
      yy = month[:4]
      mms,mme=str(int(month[4:])),str(int(month[4:]))
      dds,dde = 1,31
      # print(dd_s,dd_e,mm_s,mm_e)
      # sys.exit()
      self.driver.find_element_by_id(f"periodButton").click()
      # div= self.driver.find_element_by_xpath("//div[@class='interAnnualFlag1']")
      div= self.driver.find_element_by_css_selector(".selectpr.interAnnualFlag1")
      #checkbox click
      div.find_element_by_xpath("//select[@name='iniy']/option[@value='{}']".format(yy)).click()
      div.find_element_by_xpath("//select[@name='inim']/option[@value='{}']".format(mms)).click()
      div.find_element_by_xpath("//select[@name='inid']/option[@value='{}']".format(dds)).click() #ini_d
      div.find_element_by_xpath("//select[@name='endy']/option[@value='{}']".format(yy)).click()
      div.find_element_by_xpath("//select[@name='endm']/option[@value='{}']".format(mme)).click()
      div.find_element_by_xpath("//select[@name='endd']/option[@value='{}']".format(dde)).click()
      
    #download
    time.sleep(1)
    self.driver.find_element_by_xpath("//span[@id='csvdl']/img").click()
    Alert(self.driver).accept() # YESを押す
    time.sleep(5)
    # self.driver.quit()
    # time.sleep(3)
    return

if __name__ == "__main__":
  #loop setting...
  # _category_id = ["rklDataKnd3", "rklDataKnd4", "rklDataKnd5"]
  _category_id = ["rklDataKnd5"]
  _area_id = ["rkl1", "rkl2", "rkl3", "rkl4", "rkl5", "rkl6", "rkl7", "rkl8", "rkl9", "rkl10", "rkl11"]
  
  #initial setting...
  for cate in ["sfc"]:
    DAT = os.getcwd() + "\\dat2\\" + cate
    if os.path.exists(DAT):
      shutil.rmtree(DAT)
    os.makedirs(DAT, exist_ok=True)
  # subprocess.run("rm -f *.csv", cwd=download_dir, shell=True)
  # subprocess.run("rm -f *.csv", cwd=out_dir, shell=True)

    #loop start...
    month = "202105"
    ff = "11"
    driver = Sfc(datadir=DAT)
    driver.get(month=month, ff=ff)
    
    sys.exit()
    #rename and store data....
    rename_path = f".\\dat2\\sfc2\\{name}.csv"
    dl_csv = f"{DAT}\\data.csv"
    os.rename(dl_csv, rename_path)
    #debug....
    now = datetime.now()
    print(f"{now}[info]: end {month} - {ff}")



