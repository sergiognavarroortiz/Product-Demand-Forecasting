##############################
# Product Demand Forecasting #
##############################

#Cargar la data
urlloc="http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
download.file(urlloc,destfile = "C:/Users/Invest In data/Documents/TRABAJO/SMART DATA/R ELEMTOS DE AYUDA/Bike sharing system/BikeSharingDataset.zip",method = "libcurl")

#Descomprimir
unzip("C:/Users/Invest In data/Documents/TRABAJO/SMART DATA/R ELEMTOS DE AYUDA/Bike sharing system/BikeSharingDataset.zip")
dataset<-read.table("C:/Users/Invest In data/Documents/TRABAJO/SMART DATA/R ELEMTOS DE AYUDA/Bike sharing system/day.csv",sep=",",skip = 0,header = T)

#Revision de la data
str(dataset)

#Separacion de las variables climaticas para analisis
dataset_clim<-dataset[,-c(1:8,13:16)]
dataset_clim<-as.data.frame(dataset_clim)

#Grafico de correlacion entre variables climaticas
require(corrplot)
correlacion<-cor(dataset_clim)
corrplot(correlacion, method = "circle",order="hclust",type=c("lower"))   #Este grafico esta muy bien para mostrar graficamnete correlaciones

#Grafico de histogramas (Variable de prediccion)
hist(dataset$registered,col = "red",xlab = "Count of registered users")

#Esto para observar como esta distribuidos los datos de la variable de prediccion

#Remover atributos que no se usaran para la prediccion 
dataset$instant<-NULL
dataset$casual<-NULL
dataset$cnt<-NULL
dataset$yr<-NULL

#Converion de los atributos a factores
require(nnet)
season<-class.ind(dataset$season)   #Es como extraer el atributo del dataset y crea una matriz de cada variable categorica
colnames(season)=c("spring","summer","winter","fall")   #Esto agrega a un valor numerico una etiqueta de lo que significa su factor
head(season)

month<-class.ind(dataset$mnth)
colnames(month)=c("jan","feb","mar","april","may","jun","jul","aug","sep","oct","nov","dec")

weekday<-class.ind(dataset$weekday)
colnames(weekday)=c("sun","mon","tue","wed","thu","fri","sat")

weather<-class.ind(dataset$weather)
colnames(weather)=c("clear","mist","light")

#Creacion de Objetos de R separados para el resto de los atributos
holiday<-as.data.frame(dataset$holiday)
colnames(holiday)=c("holiday")

workingday<-as.data.frame(dataset$workingday)
colnames(workingday)=c("workingday")

temp<-as.data.frame(dataset$temp)
colnames(temp)=c("temp")

atemp<-as.data.frame(dataset$atemp)
colnames(atemp)=c("atemp")

hum<-as.data.frame(dataset$hum)
colnames(hum)=c("hum")

windspeed<-as.data.frame(dataset$windspeed)
colnames(windspeed)=c("windspeed")

#Unir todos los objetos de R en un solo data frame
sample<-cbind(season,month,holiday,weekday,workingday,weather,temp,atemp,hum,windspeed)

#Normalizar los atributos
sample<-scale(sample)

#Combinar las variables de entrenamiento con la variable de prediccion
target<-as.matrix(dataset$registered)
colnames(target)=c("target")

data<-cbind(sample,log(target)) #Le realiza un logaritmo supongo para realizar una prediccion mas acertada
data<-as.data.frame(data)

#Preparar train set
rand_seed=2016
set.seed(rand_seed)
train<-sample(1:nrow(data),700,FALSE)

#Funcion para poder aderir todas las variables para poder realizar el ajuste
#Esto en vez de realizar a~b+c+d+e
formu<-function(y_label,x_labels){
  as.formula(sprintf("%s~%s",y_label,
                     paste(x_labels,collapse = "+")))
}

#Formula almacenada en f de a~b+c+d+e+f...
f<-formu("target",colnames(data[,-33]))
f

#Creacion de un primero modelo simple con 2 hiden layers, donde la primera solo tiene una neurona
#La segunda tiene 2 neuronas

#Creacion del primer modelo simple
require(neuralnet)
set.seed(rand_seed)
fit1<-neuralnet(f,
                data = data[train,],
                hidden = c(1,3),
                algorithm = "rprop+",     #algorithm puede ser tambien rprop-,sag,slr. Donde rprop+ y rprop- refer to resilence backpropagation with and without weight backtracking; and sag refers to the smallest absolute derivate, asnd slr(smallest learning rate) refer to the modified globally converget algorithm often called grprop
                err.fct="sse",         #Funcion de error se puede seleccionar entre: sum of squared errors and cross-entropy "ce"
                act.fct = "logistic",    #La funcion de activacion puede ser: logistica (sigmoid) o tanh 
                threshold = 0.01,
                rep=1,
                linear.output = TRUE)      #linear.output=TRUE es para problemas de regresion y FALSE para problemas de clasificacion

#Para medir el desempeño del modelo se usara MSE and R-squared for a perfect fit the MSE will equal zero and
#R-Squared equal 1. 

#Medicion de la puntuacion
scores1<-compute(fit1,data[train,1:32])
pred1<-scores1$net.result

y_train=data[train,33]
require(Metrics)

round(mse(pred1,y_train),4)

round(cor(pred1,y_train)^2,4)

#Creacion de un modelo alternativo
set.seed(rand_seed)
fit2<-neuralnet(f,data = data[train,],
                hidden = c(5,6),
                algorithm = "rprop+",
                err.fct = "sse",
                act.fct = "logistic",
                threshold = 0.01,
                rep = 1,
                linear.output = TRUE)

#Puntuaje del segundo modelo
scores2<-compute(fit2,data[train,1:32])
pred2<-scores2$net.result

round(mse(pred2,y_train),4)

round(cor(pred2,y_train)^2,4)

#Creacion de un tercer modelo con un mayor numero de repeticiones
set.seed(rand_seed)
fit3<-neuralnet(f,
                data = data[train,],
                hidden = c(5,6),
                algorithm = "rprop+",
                err.fct = "sse",
                act.fct = "logistic",
                threshold = 0.01,
                rep = 10,
                linear.output = TRUE)

##Puntajes para las diferentes repeticiones de los modelos fit3
#Modelo 1
scores3<-compute(fit3,data[train,1:32])
pred3_1=as.data.frame(fit3$net.result[1])
round(mse(pred3_1[,1],y_train),4)

round(cor(pred3_1[,1],y_train)^2,4)

#Modelo 2
pred3_2<-as.data.frame(fit3$net.result[2])
round(mse(pred3_2[,1],y_train),4)

round(cor(pred3_2[,1],y_train)^2,4)

#Grafico de comportamiento del los diferentes modelos
pred3_1<-as.data.frame(fit3$net.result[1])
pred3_2<-as.data.frame(fit3$net.result[2])
pred3_3<-as.data.frame(fit3$net.result[3])
pred3_4<-as.data.frame(fit3$net.result[4])
pred3_5<-as.data.frame(fit3$net.result[5])
pred3_6<-as.data.frame(fit3$net.result[6])
pred3_7<-as.data.frame(fit3$net.result[7])
pred3_8<-as.data.frame(fit3$net.result[8])
pred3_9<-as.data.frame(fit3$net.result[9])
pred3_10<-as.data.frame(fit3$net.result[10])

r1<-round(cor(pred3_1[,1],y_train)^2,4)
r2<-round(cor(pred3_2[,1],y_train)^2,4)
r3<-round(cor(pred3_3[,1],y_train)^2,4)
r4<-round(cor(pred3_4[,1],y_train)^2,4)
r5<-round(cor(pred3_5[,1],y_train)^2,4)
r6<-round(cor(pred3_6[,1],y_train)^2,4)
r7<-round(cor(pred3_7[,1],y_train)^2,4)
r8<-round(cor(pred3_8[,1],y_train)^2,4)
r9<-round(cor(pred3_9[,1],y_train)^2,4)
r10<-round(cor(pred3_10[,1],y_train)^2,4)

rt<-rbind(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10)
plot(rt)
View(rt)

#Los modelos r8,r9 y r10 son los que no pudieron ser ajustados, por tanto aparecen con error

#El modelo que mejor resultado presento fue el segundo modelo con un valor de R-square de 0.8509
#Recordemos que mientras mas cercano a 1 mejor

#Supongamos que se usa el modelo 2, que fue el que mejor desempeño tuvo con los valores de la data de entrenamiento

y_test=data[-train,33]
scores_test2<-compute(fit2,data[-train,1:32])
pred_test2=scores_test2$net.result

round(mse(pred_test2,y_test),4)

round(cor(pred_test2,y_test),4)

#Los valores de MSE and R squared son terribles, claramente un horrible caso de sobreajuste
#Over-fitting es ajustar el modelo de una forma tan perfecta para el grupo de entrenamiento y por tanto no se puede usar para los datos de prueba porque no puede capturar la generalizacion de los datos
#Underfitting es cuando se tiene un modelo demaciado general y por tanto tampoco es bueno para modelar de forma adecuada data de prueba

#Ver como el primer modelo funciona en los datos de prueba
scores_test1<-compute(fit1,data[-train,1:32])
pred_test1=scores_test1$net.result

round(mse(pred_test1,y_test),4)
#Resultado de 0.1413
round(cor(pred_test1,y_test),4)
#Resultado de 0.7709

#Los resultados son mucho mejores que los obtenidos por el modelo fit2

#Grafico de los valores reales y ajustados
x<-c(0,31)
y<-c(0,7000)
plot(x,y,col="white",xlab = "Test Examples(Days)",ylab = "Numbers Registered")
lines(exp(pred_test1),col="red")
lines(exp(y_test),col="blue")





