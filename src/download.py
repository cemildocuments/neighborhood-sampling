"""Download checksum-pinned public cell tables from the primary releases."""
from pathlib import Path
import requests,hashlib,json,argparse,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
def fetch(item):
 p=ROOT/item['file'];p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']:return p.name+' verified'
 for attempt in range(3):
  try:
   response=requests.get(item['url'],timeout=(30,180));response.raise_for_status();data=response.content
   if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Checksum mismatch: '+p.name)
   temp=p.with_suffix(p.suffix+'.part');temp.write_bytes(data);temp.replace(p);return p.name+' downloaded'
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for message in pool.map(fetch,json.load(open(ROOT/'metadata/input_manifest.json'))):print(message,flush=True)
