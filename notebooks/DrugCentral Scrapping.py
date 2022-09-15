from bs4 import BeautifulSoup
import requests
import pandas as pd


print("Start")
print("Masukan Uniprot ID:")
ID = input()
print("Save file name as:")
fname = input()

def ScrappingDC():
  url = 'https://drugcentral.org/?q='+ID
  data = requests.get(url).text
  soup = BeautifulSoup(data,"html.parser")

  num = soup.p.get_text()
  num = num.split(": ")
  num = num[1].replace(" ","")
  num = int(num)
  tabs = []
  for i in range(num*2):
    link = soup.table.find_all("a", href=True)[i]
    link = link.get_text()
    link = link.replace("\n","")
    tabs.append(link)
  while("" in tabs) :
      tabs.remove("")

  CompID = []
  for i in tabs:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{i}/cids/JSON"

    r = requests.get(url)
    r.raise_for_status()
    response = r.json()
    if "IdentifierList" in response:
        cid = response["IdentifierList"]["CID"][0]
        CompID.append(cid)
    else:
        raise ValueError(f"Could not find matches for compound: {i}")
  df = pd.DataFrame(CompID, columns = ["Compund"])
  saveas = fname+'.csv'
  df.to_csv(saveas,index=False)

ScrappingDC()