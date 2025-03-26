import pandas as pd 

def read_data(a, b, sample_rate, hip=0):


    # 0 : hip translation, no hip joint
    # 1 : no hip translation, with hip joint
    # 2 : hip translation, with hip joint
    # 3 : hip translation, with difference of hip joints

    # Read files
    df = a
    df2 = b

    # Drop time columns

    df = df.drop(df.columns[0], axis=1)  
    df2 = df2.drop(df2.columns[0], axis=1)  

    # Sample down to 24 fps

    df = df.iloc[::int(1/sample_rate)]
    df2 = df2.iloc[::int(1/sample_rate)]

    # Do the hip translation

    if hip == 0 or hip == 2 or hip == 3:
        subx = [a for a in range(3,78) if a%3 == 0]
        suby = [a for a in range(3,78) if a%3 == 1]
        subz = [a for a in range(3,78) if a%3 == 2]
        for s in subx:
            df.iloc[:, s] -= df.iloc[:, 0]
            df2.iloc[:, s] -= df2.iloc[:, 0]
        for s in suby:
            df.iloc[:, s] -= df.iloc[:, 1]
            df2.iloc[:, s] -= df2.iloc[:, 1]
        for s in subz:
            df.iloc[:, s] -= df.iloc[:, 2]
            df2.iloc[:, s] -= df2.iloc[:, 2]
        
    # If we want the distance between caregiver and carereceiver given as input to the model instead of global position
    if hip == 3 :
        sub = df.iloc[:,0:3].values - df2.iloc[:,0:3].values
        df.iloc[:,0:3] = sub
        df2.iloc[:,0:3] = sub

    if hip == 0:
         # If we do not want to feed the model the global hip position information, we drop the the 0'th joints XYZ coordinates
        idx_toremove = [0,1,2,5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]
    else:
        idx_toremove = [5*3,5*3+1,5*3+2,10*3,10*3+1,10*3+2,17*3,17*3+1,17*3+2,22*3,22*3+1,22*3+2,25*3,25*3+1,25*3+2]

    df = df.drop(df.columns[idx_toremove], axis=1)  
    df2 = df2.drop(df2.columns[idx_toremove], axis=1)  
    return df, df2
