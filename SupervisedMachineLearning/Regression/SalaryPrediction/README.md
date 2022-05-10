In this project we aimed to compare the salaries of data scientist and software engineers by taking in the survey data from levels.fyi

The data comprises of details such as:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 62642 entries, 0 to 62641
Data columns (total 29 columns):
 #    Column                  Non-Null Count  Dtype         
---  ------                   --------------  -----         
 0   timestamp                62642 non-null  datetime64[ns]
 1   company                  62637 non-null  object        
 2   level                    62523 non-null  object        
 3   title                    62642 non-null  object        
 4   totalyearlycompensation  62642 non-null  int64         
 5   location                 62642 non-null  object        
 6   yearsofexperience        62642 non-null  float64       
 7   yearsatcompany           62642 non-null  float64       
 8   tag                      61788 non-null  object        
 9   basesalary               62642 non-null  int64         
 10  stockgrantvalue          62642 non-null  float64       
 11  bonus                    62642 non-null  float64       
 12  gender                   43102 non-null  object        
 13  otherdetails             40134 non-null  object        
 14  cityid                   62642 non-null  int64         
 15  dmaid                    62640 non-null  float64       
 16  rowNumber                62642 non-null  int64         
 17  Masters_Degree           62642 non-null  int64         
 18  Bachelors_Degree         62642 non-null  int64         
 19  Doctorate_Degree         62642 non-null  int64         
 20  Highschool               62642 non-null  int64         
 21  Some_College             62642 non-null  int64         
 22  Race_Asian               62642 non-null  int64         
 23  Race_White               62642 non-null  int64         
 24  Race_Two_Or_More         62642 non-null  int64         
 25  Race_Black               62642 non-null  int64         
 26  Race_Hispanic            62642 non-null  int64         
 27  Race                     22427 non-null  object        
 28  Education                30370 non-null  object        
dtypes: datetime64[ns](1), float64(5), int64(14), object(9)

We tried our best to identify all the discrepenies, perform the cleaning and try to feed the quality data to our model. 
We paid attention to make a note of the impact caused by the data cleaning on the distribution of our data.

<img width="1016" alt="image" src="https://user-images.githubusercontent.com/31846843/167707969-70e2eb8a-1989-47f7-af96-c307ea301ebf.png">

<img width="931" alt="image" src="https://user-images.githubusercontent.com/31846843/167708025-e51153c8-6d0e-4f94-9ead-fa010041940b.png">

<img width="939" alt="image" src="https://user-images.githubusercontent.com/31846843/167708098-64d25aa6-0e3b-4049-a811-8585f8ba74c4.png">

