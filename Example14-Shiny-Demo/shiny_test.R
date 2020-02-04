install.packages('shiny') 

library('cdsw')
library('shiny')
library('parallel')

mcparallel(runApp(host="0.0.0.0", port=8080, launch.browser=FALSE,
 appDir="/home/cdsw/app", display.mode="auto"))

service.url <- paste("http://", Sys.getenv("CDSW_ENGINE_ID"), ".",
Sys.getenv("CDSW_DOMAIN"), sep="")
Sys.sleep(5)

iframe(src=service.url, width="640px", height="480px")
