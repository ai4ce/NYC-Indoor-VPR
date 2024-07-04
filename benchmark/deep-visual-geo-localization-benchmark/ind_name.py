# fd=open("test_database.txt")
# fq=open("test_queries.txt")
# fdl=fd.readlines()
# fql=fq.readlines()
# f=open("anyloc.txt")
# f1=open("results/anyloc_name.txt","w")
# lines=f.readlines()
# for i in range(1,len(lines)):
#     s=lines[i].strip().split()
#     f1.write(fql[int(s[0])].strip()+" "+fdl[int(s[1])].strip()+"\n")
import math

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# f=open("results/resnet_name.txt")
# f1=open("results/resnet_eval.txt","w")
# for line in f:
#     s=line.strip().split()
#     x0=float(s[0].split('@')[1])
#     y0=float(s[0].split('@')[2])
#     x1=float(s[1].split('@')[1])
#     y1=float(s[1].split('@')[2])
#     f1.write(s[0]+" "+s[1]+" "+str(calculate_distance((x0,y0),(x1,y1)))+"\n")

ad={}
f=open("results/anyloc_eval.txt")
for line in f:
    s=line.strip().split()
    ad[s[0]]=(s[1],float(s[2]))

cd={}
f1=open("results/cct_eval.txt")
for line in f1:
    s=line.strip().split()
    cd[s[0]]=(s[1],float(s[2]))

cod={}
f2=open("results/cosplace_eval.txt")
for line in f2:
    s=line.strip().split()
    cod[s[0]]=(s[1],float(s[2]))

md={}
f3=open("results/mixvpr_eval.txt")
for line in f3:
    s=line.strip().split()
    md[s[0]]=(s[1],float(s[2]))
  
rd={}
f4=open("results/resnet_eval.txt")
for line in f4:
    s=line.strip().split()
    rd[s[0]]=(s[1],float(s[2]))

ad_rate={}
for k,v in cod.items():
    m=k.split('@')[1][1]
    if m not in ad_rate:
        ad_rate[m]=[0,0]
    if v[1]<=25:
        ad_rate[m][0]+=1
    else:
        ad_rate[m][1]+=1

a=[]
for i in range(0,9):
    a.append(round(ad_rate[str(i)][0]/(ad_rate[str(i)][0]+ad_rate[str(i)][1]),2))
# print(a)
# common_keys = set(ad.keys()) & set(cd.keys()) & set(rd.keys()) & set(md.keys())
# num_common_keys = len(common_keys)
# print(num_common_keys)

# for k,v in md.items():
#     if float(k.split('@')[1])<8000 or float(k.split('@')[1])>9000:
#         continue
#     if v[1]<5 and (k in ad) and (k in cd) and (k in rd):
#         print(k, ad[k],cd[k],rd[k],md[k],sep="\n")
#         break
        
# k="/mnt/data/nyc_indoor/indoor/images/test/queries/@08115.45@00140.84@620@.jpg"
# print(k, ad[k],cd[k],rd[k],md[k],sep="\n")
# for k,v in md.items():
#     if float(k.split('@')[1])<500:
#         continue
#     if v[1]<5 and (k in ad) and (k in cd) and (k in rd) and (ad[k][1]>20):
#         print(k, ad[k],cd[k],rd[k],md[k],sep="\n")
#         break
        
# for k,v in md.items():
#     if float(k.split('@')[1])<1500 or (float(k.split('@')[1])>8000 and float(k.split('@')[1])<9000):
#         continue
#     if v[1]<5 and (k in ad) and (k in cd) and (k in rd) and (ad[k][1]>20 or cd[k][1]>20 or rd[k][1]>20):
#         print(k, ad[k],cd[k],rd[k],md[k],sep="\n")
#         break

s=["/mnt/data/nyc_indoor/indoor/images/test/queries/@08115.45@00140.84@620@.jpg"]

kn=""
for q in s:
    maxd=9999
    coord=(float(q.split('@')[1]),float(q.split('@')[2]))
    for k,v in cod.items():
        c1=(float(k.split('@')[1]),float(k.split('@')[2]))
        if calculate_distance(coord,c1)<maxd:
            maxd=calculate_distance(coord,c1)
            kn=k
    print(cod[kn])