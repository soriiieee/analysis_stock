import os, sys
import time
import pandas as pd
import shutil

from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
options = Options()
#init 
options.add_argument('--disable-gpu')
options.add_argument('--disable-extensions')
options.add_argument('--proxy-server="direct://"')
options.add_argument('--proxy-bypass-list=*')
options.add_argument('--start-maximized')

import subprocess
# import warnings 
# warnings.simple

"""
参照記事
〇webdriver(chrome)の使い方
https://tanuhack.com/selenium/#WebDriver
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
  def __init__(self,datadir,month,name="TOUKYOU"):
    #firefox setting
    #---driverpath(old)---------
    # driver_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\geckodriver.exe"
    # driver_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
    # driver_path="C:\\Users\\1119041\\Desktop\\geckodriver-v0.27.0-win64\\chromedriver.exe"
    
    #---driverpath(laetest))---------
    # https://pypi.org/project/chromedriver-binary/#history で最新verをinstallできる
    # pip install chromedriver-binary==91.0.4472.19.0 -> 事前にこれやらないと勝手に更新される
    # pip install chromedriver-binary==96.0.4664.18.0
    # pip install chromedriver-binary==95.0.4638.10.0
    driver_path="C:\\Users\\1119041\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\chromedriver_binary\\chromedriver.exe" #2021.09.20 更新-selenium対応-
    
    # 
    # print(os.path.exists(driver_path))
    # sys.exit()
    # self.driver = webdriver.Firefox(firefox_profile=fp, executable_path=driver_path,options=options)
    # self.driver.implicitly_wait(10)
    self.driver = webdriver.Chrome(executable_path=driver_path, chrome_options=options)
    self.driver.implicitly_wait(10) # 秒
    self.name = name
    self.month = month
    self.tbl = self.list_tbl()
    # print(self.tbl.head())
    # sys.exit()
    # sys.exit()
    #chrome の設定方法
# ファイルのデフォルトの保存先を変更する
    self.driver.command_executor._commands['send_command'] = (
        'POST',
        '/session/$sessionId/chromium/send_command'
    )
    params = {
        'cmd': 'Page.setDownloadBehavior', 
        'params': {
            'behavior': 'allow',
            'downloadPath': datadir
        }
    }
    self.driver.execute('send_command', params=params)
    # sys.exit()
    # shutil.rmtree(datadir)
    # os.makedirs(datadir, exist_ok=True)
    # self.month = month
    # self.ff = ff
  def get(self):
    ff,scode = self.get_scode()
    self.driver.get("https://www.data.jma.go.jp/gmd/risk/obsdl/index.php")
    # WebDriverWait(self.driver, 5).until(lambda d: len(d.window_handles) > 1)
    # handles = self.driver.window_handles
    # sys.exit()
    # self.driver.switch_to.window(handles[0])
    """地点検索"""
    if 1:
      self.driver.find_element_by_id(f"pr{ff}").click()
      # v = self.driver.find_element_by_id("stationArea").find_element_by_name("stid")
      element_text = "//div[input[@name='stid'][@value='s{}']]".format(scode)
      e = self.driver.find_element_by_xpath(element_text)
      time.sleep(1)
      e.click()
      time.sleep(1)
    
    """要素選定"""
    if 1:
      self.driver.find_element_by_id(f"elementButton").click()
      #1jikann 毎
      # time.sleep(1)
      self.driver.find_element_by_xpath("//input[@name='aggrgPeriod'][@value='9'][@type='radio']").click()
      # time.sleep(1)
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
      yy = self.month[:4]
      mms,mme=str(int(self.month[4:])),str(int(self.month[4:]))
      dds,dde = 1,self.dd()
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
    # Alert(self.driver).accept() # YESを押す
    time.sleep(3)
    self.driver.quit()
    # time.sleep(3)
    return

  def list_tbl(self,LIST=None):
    # path = "C:\\Users\\1119041\\Desktop\\tmp\\20210614\\total_amedas\\total_ame.csv"
    path = "C:\\Users\\1119041\\Desktop\\sori_workspace\\selenium\\tbl\\list_sokuhou.csv"
    tbl = pd.read_csv(path)
    # tbl = tbl[(tbl["scode"] != 9999)&(tbl["cate"]=='官')].sort_values("scode")
    tbl = tbl.drop_duplicates(subset=["code"])
    # print(tbl.head())
    # sys.exit()
    if LIST:
      tbl = tbl.loc[tbl["name"].isin(_list),:]
    else:
      pass
    return tbl
  
  def get_scode(self):
    scode = self.tbl.loc[self.tbl["name"]==self.name,"scode"].values[0]
    code = self.tbl.loc[self.tbl["name"]==self.name,"code"].values[0]
    ff = str(code)[:2]
    scode = str(scode)
    
    if self.name == "IRIOMOTEJIMA" or self.name == "ISHIGAKIJIMA":
      ff = "91"
    return ff,scode
  
  def dd(self):
    if self.month=="202105":
      return 31
    if self.month=="202106":
      return 30
    if self.month=="202107":
      return 31
    if self.month=="202108":
      return 31
    if self.month=="202109":
      return 30
    if self.month=="202110":
      return 31
    if self.month=="202111":
      return 30
    if self.month=="202112":
      return 31
    if self.month=="202201":
      return 31
    if self.month=="202202":
      return 28
    if self.month=="202203":
      return 31
    if self.month=="202204":
      return 30


def list_tbl(LIST=None):
  path = "C:\\Users\\1119041\\Desktop\\tmp\\20210614\\total_amedas\\total_ame.csv"
  tbl = pd.read_csv(path)
  tbl = tbl[(tbl["scode"] != 9999)&(tbl["cate"]=='官')].sort_values("scode")
  tbl = tbl.drop_duplicates(subset=["code"])
  if LIST:
    tbl = tbl.loc[tbl["name"].isin(LIST),:]
  else:
    pass
  return tbl



if __name__ == "__main__":
  #loop setting...
  LIST=["ASAHIKAWA","SAPPORO","AOMORI","SENDAI","TOYAMA","NAGANO","TOUKYOU","MIYAKEJIMA","HACHIJOUJIMA","KYOUTO","OOSAKA","FUKUOKA","NAHA","IRIOMOTEJIMA","ISHIGAKIJIMA"]
  LIST=["KYOUTO"]
  # month = "202106" #2021.07.10
  # month = "202107" #2021.08.19
  # month = "202108" #2021.09.20
  # month = "202109" #2021.10.18
  # month = "202110" #2021.11.15
  month = "202111" #2021.12.13
  cate = "sfc"
  
  log_f = "./get_jma_ch.log"
  subprocess.run("rm -f {}".format(log_f), shell = True)
  
  #initial setting...
  DAT = os.getcwd() + "\\dat2\\" + cate
  OUT = os.getcwd() + "\\dat2\\" + cate+"2"
  
  if os.path.exists(DAT):
    try:
      shutil.rmtree(DAT)
    except:
      pass
  if os.path.exists(OUT):
    try:
      shutil.rmtree(OUT)
    except:
      pass
  os.makedirs(DAT, exist_ok=True)
  os.makedirs(OUT, exist_ok=True)
  # init file remove
  try:
    subprocess.run(f"rm {DAT}\\*.csv",shell=True) #init_rm
    subprocess.run(f"rm {OUT}\\*.csv",shell=True) #init rm
  except:
    pass
  # subprocess.run(f"rm {OUT}\\*.csv",shell=True)
  # sys.exit()
  # subprocess.run("rm -f *.csv", cwd=out_dir, shell=True)
  for name in LIST:
    #get...
    subprocess.run("rm data.csv", cwd=DAT, shell=True)
    driver = Sfc(datadir=DAT,month=month,name=name)
    driver.get()
  
    # log....
    with open(log_f, "+a") as f:
      dl_csv = f"{DAT}\\data.csv"
      if os.path.exists(dl_csv):
        flg=0
        rename_path = f"{OUT}\\{name}.csv"
        os.rename(dl_csv, rename_path)
      else:
        flg=1
      now = datetime.now()
      text = f"{now}[end] {name} {month} FLG={flg}\n"
      f.write(text)
      
      # sys.exit()



