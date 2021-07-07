# Accident Prediction Model
# By : Jill Bhatt, Trupal Chaudhary, Kavan Dalwadi
#
#Divided into 2 Modules : Insights and Predictions


#---Libraries--# 

  library(shiny)
  library(shinydashboard)
  library(DT)
  library(reshape2)
  library(ggplot2)
  library(keras)
  library(tensorflow)
  library(ggplot2)
  
  #---Libraries END--# 
  
  #--CSV--#
  
  alc_data = read.csv("CSV/alcohol.csv")
  alc_data.m <- melt(alc_data, id.vars="ï..Year")
  
  spd_data = read.csv("CSV/speeding.csv")
  spd_data.m <- melt(spd_data, id.vars="ï..Year")
  
  ovl_data = read.csv("CSV/overloading.csv")
  ovl_data.m <- melt(ovl_data, id.vars="ï..Year")
  
  ovc_data = read.csv("CSV/overcrowding.csv")
  ovc_data.m <- melt(ovc_data, id.vars="ï..Year")
  
  wrg_data = read.csv("CSV/wrongside.csv") 
  wrg_data.m <- melt(wrg_data, id.vars="ï..Year")
  
  crs_data = read.csv("CSV/crosswalk.csv") 
  crs_data.m <- melt(crs_data, id.vars="ï..Year")
  
  brd_data = read.csv("CSV/bridges.csv") 
  brd_data.m <- melt(brd_data, id.vars="ï..Year")
  
  urb_data = read.csv("CSV/urbanarea.csv")
  urb_data.m <- melt(urb_data, id.vars="ï..Year")
  
  rur_data = read.csv("CSV/ruralarea.csv")
  rur_data.m <- melt(rur_data, id.vars="ï..Year")
  
  age_data = read.csv("CSV/ageofveh.csv") 
  age_data.m <- melt(age_data, id.vars="ï..Year")
  
  drw_data = read.csv("CSV/drowsiness.csv") 
  drw_data.m <- melt(drw_data, id.vars="ï..Year")
  
  #--CSV END--#
  
  
  #--UI--#
  
  ui <- dashboardPage(skin = "red",
          dashboardHeader(title = "Accident Prediction Model", titleWidth = 300),
          dashboardSidebar(width = 300,
            sidebarMenu(
              menuItem("Insights", tabName = "iris", icon = icon("exclamation-triangle")),
              menuItem("Prediction", tabName = "cars", icon = icon("car-crash"))
          )),
          dashboardBody(
              tabItems(
              tabItem("iris",
                  box(plotOutput("correlation_plot"), width=12, height=400),
                  box(selectInput("features", "Factor:",
                    c("Alcohol", "Speeding", "Overloading" , "Overcrowding", "Driving Wrong Side", "Crosswalk" , "Bridges", "Urban Area" , "Rural Area", "10+ Age of Vehicle", "Drowsiness")
                  ),
                  width = 3, 
                )
            ),
              tabItem("cars", 
                  box(plotOutput("prediction_plot"), width = 12, height = 400),
                  box(selectInput("predfeat", "Factor:",
                    c("Alcohol", "Speeding", "Overloading", "Overcrowding")
                  ), 
                  width = 6,
              )
            )
          )
        )   
      )
      
  
    
  #--Server Function--#
  
  server <- function(input, output){
    
      #--Insights Module--# //Jill Bhatt
    
      output$correlation_plot <- renderPlot(height = 275,{
          x <- switch(input$features,
                   "Alcohol" = alc_data.m,
                   "Speeding" = spd_data.m,
                   "Overloading" = ovl_data.m ,
                   "Overcrowding" = ovc_data.m,
                   "Driving Wrong Side" = wrg_data.m, 
                   "Crosswalk"= crs_data.m, 
                   "Bridges"=brd_data.m, 
                   "Urban Area" = urb_data.m,
                   "Rural Area" = rur_data.m,
                   "10+ Age of Vehicle" = age_data.m,
                   "Drowsiness" = drw_data.m)
                   
      
      ggplot(x , aes(ï..Year, value)) +
          geom_bar(aes(fill = variable), width = 0.4, position = position_dodge(width=0.5), stat="identity") +  
          theme(legend.position="top", legend.title = element_blank(),axis.title.x=element_blank(),axis.title.y=element_blank()) +
          geom_text(aes(label=value), position=position_dodge(width=3),hjust="center",vjust ="center", size= 3)+
          scale_x_continuous("ï..Year", labels = as.character(Year), breaks = Year)
      })
      
      #--Insights Module END--#
      
      #--Prediction Module--# // Trupal and Kavan
    
      output$prediction_plot <- renderPlot(height = 275,{
          x <- switch(input$predfeat,
                      "Alcohol" = "CSV/alcohol.csv",
                      "Speeding" = "CSV/speeding.csv",
                      "Overloading" = "CSV/overloading.csv" ,
                      "Overcrowding" = "CSV/overcrowding.csv")
      
      data <- read.csv(x)
      Year <- data$ï..Year
      Value <- data$Total
      
      df = data.frame(Year, Value)
      Series = df$Value
      
      diffed = diff(Series, differences = 1)    # Transforming data to stationary 
      head(diffed)                              # Difference between two consecutive values in the series
      
      
      #--Lag Function-------------------------- LSTM expects the data to be in a supervised learning mode (target variable Y and predictor X) 
      lags <- function(x, k){
        
          lagged =  c(rep(NA, k), x[1:(length(x)-k)])
          DF = as.data.frame(cbind(lagged, x))
          colnames(DF) <- c( paste0('x-', k), 'x')
          DF[is.na(DF)] <- 0
          return(DF)
      }
      supervised = lags(diffed, k=1)
      
      #--Lag Function END--#
      
      #--Normalization------------------------- For default activation function for LSTM is sigmoid function whose range is [-1, 1]
      
      N = nrow(supervised)
      n = round(N *0.66, digits = 0)
      train = supervised[1:n, ]
      test  = supervised[(n+1):N,  ]
      
      
      normalize <- function(train, test, feature_range = c(0, 1)) {
        
          x = train
          fr_min = feature_range[1]
          fr_max = feature_range[2]
          std_train = ((x - min(x) ) / (max(x) - min(x)  ))
          std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
        
          scaled_train = std_train *(fr_max -fr_min) + fr_min
          scaled_test = std_test *(fr_max -fr_min) + fr_min
        
          return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
        
      }
      
      #--Normalization END--#
      
      #--Invert Function------------------------ For reverting the predicted values to the original scale
      
      
      
      inverter = function(scaled, scaler, feature_range = c(0, 1)){
        
          min = scaler[1]
          max = scaler[2]
          n = length(scaled)
          mins = feature_range[1]
          maxs = feature_range[2]
          inverted_dfs = numeric(n)
        
          for( i in 1:n){
            
              X = (scaled[i]- mins)/(maxs - mins)
              rawValues = X *(max - min) + min
              inverted_dfs[i] <- rawValues        
          }
          return(inverted_dfs)
      }
      
      #--Invert Function END--#
      
      
      Scaled = normalize(train, test, c(-1, 1))
      
      y_train = Scaled$scaled_train[, 2]
      x_train = Scaled$scaled_train[, 1]
      
      y_test = Scaled$scaled_test[, 2]
      x_test = Scaled$scaled_test[, 1]
      
      
      #--LSTM Model--#
      
      dim(x_train) <- c(length(x_train), 1, 1)
      dim(x_train)
      
      X_shape2 = dim(x_train)[2]
      X_shape3 = dim(x_train)[3]
      
      batch_size = 1
      units = 1
      
      model <- keras_model_sequential() 
      model%>%
          layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
          layer_dense(units = 1)
      
      
      
      model %>% compile(
          loss = 'mean_squared_error',
          optimizer = optimizer_adam( lr= 0.02 , decay = 1e-6 ),  
          metrics = c('accuracy')
      )
      
      
      summary(model)
      
      
      #--Fitting Model--#
      
      Epochs = 25
      nb_epoch = Epochs
      
      for(i in 1:nb_epoch ){
        
          model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
          model %>% reset_states()
        
      }
      
      #--Fitting Model END--#
      
     
      L = length(x_test)
      dim(x_test) = c(length(x_test), 1, 1)
      
      scaler = Scaled$scaler
      
      predictions = numeric(L)
   
      for(i in 1:L){
        
          X = x_test[i , , ]
          dim(X) = c(1,1,1)
          yhat = model %>% predict(X, batch_size=batch_size)
          yhat = inverter(yhat, scaler,  c(-1, 1))
          yhat  = yhat + Series[(n+i)] 
        
          predictions[i] <- yhat
      }
      
      #--LSTM Model END--#
      
      year2<- c(2017,2018,2019)
      df2$Year<-year2
      df2$predictions <- predictions
      df2 <- df2[c("Year", "predictions")]
      
      
      #--Graphs--#
      ggplot()+
          geom_line(data = df, aes(x = Year, y = Value),color = "#00AFBB")+
          geom_line(data = df2, aes(x= Year, y = predictions), color = "red")+
          geom_text(aes(x=df2$Year, y=df2$predictions, label= df2$predictions), vjust = "inward", hjust = "inward", show.legend = FALSE, size = 4)
      
      })
      
      #--Graphs--#
      
      #--Prediction Module END--#
      
      #--Server Function END--#
    
  }
  
  shinyApp(ui, server)