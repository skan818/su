# Script will open an embryoscope database and extract out a sequential set of jpg files for each embryo across time within
# the plane of focus (0).  File to to be used to train max blastocyst size.

library(RSQLite)
setwd("/data/embryo/")
a<-list('/data/embryo/pdb/D2019.04.23_S00442_I0831_D.pdb')

writeJpgs<-function(embryoscopeDB,base_dir){

convertTime <- function(x){
  Hour = (x * 24) %% 24 #For x>1, only the value after decimal will be considered
  Minutes = (Hour %% 1) * 60
  Seconds = (Minutes %% 1) * 60
  
  hrs = ifelse (Hour < 10, paste("0",floor(Hour),sep = ""), as.character(floor(Hour)))
  mins = ifelse (Minutes < 10, paste("0",floor(Minutes),sep = ""), as.character(floor(Minutes)))    
  secs = ifelse (Seconds < 10, paste("0",round(Seconds,0),sep = ""), as.character(round(Seconds,0)))
  
  return(paste(hrs, mins, secs, sep = ":"))
}

rawToJpeg <- function(pic_data,file.name) {
  f = file(file.name, "wb")           # OPEN FILE CONNECTION
  # modify pic data
  pic.mod<-c(as.raw(pic_data[[1]]),as.hexmode("ff"),as.hexmode("d9"))
  writeBin(as.raw(pic_data[[1]]), con = f, useBytes=TRUE)      # TRANSFER RAW DATA
  close(f)                                        # CLOSE FILE CONNECTION
}

# Connect to DB
con <- dbConnect(SQLite(), embryoscopeDB)
# Extract out all in focus images
img2<-dbGetQuery(con, "SELECT * FROM IMAGES WHERE Focal == '0'")
# Grab Fertilization time from file to handle multiple files
read.q<-paste0("SELECT Val FROM GENERAL WHERE Par =  'Fertilization'")
fert.time<-dbGetQuery(con,read.q)

# For each embryo compute time since start

start.time<-as.numeric(fert.time)


for (j in unique(img2$Well)){
  print(j)
  tmp.dat<-img2[img2$Well==j,]

  new.time<-floor((tmp.dat$Time-start.time) * 1440) # convert time to minutes since start
  name.tmp<-unlist(strsplit(basename(embryoscopeDB),"_P.pdb"))
  ename<-paste0(name.tmp,"_E",j)
  ename<-gsub("\\.","_",ename)
  dir.create(file.path(base_dir,ename),showWarnings = F)
  print(paste0("Writing files for ",ename))
  for(k in 1:nrow(tmp.dat)){
    fname<-paste0(ename,"_",new.time[k],".jpg")
    #print(paste0("Writing file ",fname," (",k," of ",nrow(tmp.dat),")"))
    rawToJpeg(tmp.dat[k,5],file.path(base_dir,ename,fname))
  
  }
}
dbDisconnect(con)
}

base_dir<-"/data/embryo/day_6"
#writeJpgs(a[1],base_dir)
library(parallel)
mclapply(a,function(x) writeJpgs(x,base_dir),mc.cores=6)
