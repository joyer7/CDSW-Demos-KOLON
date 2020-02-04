import ibis
import os
ibis.options.interactive = True

#host where httpfs lives
hdfs_host = os.getenv('HDFS_HOST', 'ip-10-0-0-168.us-west-2.compute.internal')
hdfs = ibis.hdfs_connect(host=hdfs_host, port=14000,
                         auth_mechanism='GSSAPI', verify=True, use_https=False)
hdfs.ls('/tmp')

#host where impala daemon lives
impala_host = os.getenv('IMPALA_HOST', 'ip-10-0-0-150.us-west-2.compute.internal')
con = ibis.impala.connect(host=impala_host, port=21050, 
                          database='flights', hdfs_client=hdfs,
                          auth_mechanism='GSSAPI', use_ssl=False)
con.list_tables()

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
airlines = con.table('airlines_bi_pq')
airlines.limit(10).execute()


airports = con.table('airports')
airports.limit(10).execute()

airlines_by_year = airlines.group_by(by="year").aggregate(count=airlines.count()).sort_by('year')

sns.set(font_scale=0.7)

sns.pointplot(x='year', y='count', data=airlines_by_year.execute())

airlines_by_year_carrier = airlines.group_by("carrier").aggregate(count=airlines.count()).sort_by(["year", "carrier"])

delays = airlines['year', 'carrier', 'depdelay', 'arrdelay'].mutate(gain = airlines.depdelay - airlines.arrdelay)[airlines.depdelay.notnull() & airlines.arrdelay.notnull()]['year', 'carrier', 'gain']
buckets = [-100, -10, 0, 10, 20, 30, 40, 50, 100]
bucketed = delays.gain.bucket(buckets). \
label(['-100 to -10', '-10 to 0', '0 to 10', '10 to 20', '20 to 30', '30 to 40', '40 to 50', '50 to 100']).name('bucket')
bucketed.value_counts().sort_by('bucket')

