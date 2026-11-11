import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
for item in json.loads(Path(__file__).with_name('release_assets.json').read_text()):
 f=a.directory/item['name'];h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise ValueError('SHA256 mismatch: '+f.name)
 print('PASS',f.name)
