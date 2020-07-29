import plotnine as p9
import pandas as pd

data = pd.read_csv("/data/embryo/tfrecords/yolov3-tf2/single_batch.csv")
zona_data = data[data.Class == 'zona']

plot = (p9.ggplot(data = zona_data,
mapping = p9.aes(x = 'Time', y = 'Size', color = 'ID')))
plot + p9.geom_smooth(method = 'loess', span = 0.4)
