import numpy as np


def calculate_and_include_all_features(df, window_size):
    df = calculate_and_include_distance(df)
    df = calculate_and_include_velocity(df)
    df = calculate_and_include_acceleration(df)
    df = calculate_and_include_accelerationx(df)
    df = calculate_and_include_accelerationy(df)
#     df = calculate_and_include_accelerationz(df)
    df = calculate_and_include_jerk(df)
    df = calculate_and_include_jerkx(df)
    df = calculate_and_include_jerky(df)
    df = calculate_and_include_z_speed(df)
    df = calculate_and_include_theta(df)
    df = calculate_and_include_best_fit_features(df, window_size)
    df = calculate_and_include_slope_diff(df)
    df = calculate_and_include_mean_and_var(df, window_size)
    return df


# Individual feature extraction functions

def calculate_and_include_distance(df):
    for i in range(len(df)):
        df[i]['Distance'] = np.append(0, (np.sqrt(np.power(np.diff(df[i]['DeviceCoordinate1']), 2),
                    np.power(np.diff(df[i]['DeviceCoordinate2']), 2))))
    return df


def calculate_and_include_velocity(df):
    for i in range(len(df)):
        df[i]['Velocity'] = np.sqrt(df[i]['ScenarioVelocity1'].pow(2) + df[i]['ScenarioVelocity2'].pow(2))
    return df

def calculate_and_include_z_speed(df):
    for i in range(len(df)):
        df[i]['VelocityZ'] = np.append(np.array(0), np.diff(df[i]['HeightAboveGroundLevel']) / (
                    np.diff(df[i]['ScenarioTimeStamp']) )) # np.abs(np.diff(df[i]['HeightAboveGroundLevel']) / (
                    # np.diff(df[i]['ScenarioTimeStamp']) / 1000)))
        df[i]['AccelerationZ'] = np.append(np.array(0), np.diff(df[i]['VelocityZ']) / 
                    (np.diff(df[i]['ScenarioTimeStamp'])))
        df[i]['JerkZ'] = np.append(np.array(0), np.diff(df[i]['AccelerationZ']) / 
                    (np.diff(df[i]['ScenarioTimeStamp'])))
    return df

def calculate_and_include_acceleration(df):
    for i in range(len(df)):
        df[i]['Acceleration'] = np.append(np.array(0), np.diff(df[i]['Velocity'])
                                          / (np.diff(df[i]['ScenarioTimeStamp'])))
    return df

def calculate_and_include_accelerationx(df):
    for i in range(len(df)):
        df[i]['AccelerationX'] = np.append(np.array(0), np.diff(df[i]['ScenarioVelocity1'])
                                          / (np.diff(df[i]['ScenarioTimeStamp']) ))
    return df

def calculate_and_include_accelerationy(df):
    for i in range(len(df)):
        df[i]['AccelerationY'] = np.append(np.array(0), np.diff(df[i]['ScenarioVelocity2'])
                                          / (np.diff(df[i]['ScenarioTimeStamp']) ))
    return df

def calculate_and_include_jerk(df):
    for i in range(len(df)):
        df[i]['Jerk'] = np.append(np.array(0), np.diff(df[i]['Acceleration']) / (
                    np.diff(df[i]['ScenarioTimeStamp'])))
    return df

def calculate_and_include_jerkx(df):
    for i in range(len(df)):
        df[i]['JerkX'] = np.append(np.array(0), np.diff(df[i]['AccelerationX']) / (
                    np.diff(df[i]['ScenarioTimeStamp'])))
    return df

def calculate_and_include_jerky(df):
    for i in range(len(df)):
        df[i]['JerkY'] = np.append(np.array(0), np.diff(df[i]['AccelerationY']) / (
                    np.diff(df[i]['ScenarioTimeStamp'])))
    return df

def calculate_and_include_theta(df):
    for i in range(len(df)):
        a = np.power(np.diff(df[i]['DeviceCoordinate2']),2)
        b = np.power(np.diff(df[i]['DeviceCoordinate1']),2)
        df[i]['Theta'] = np.append(np.array(0), np.arctan(np.sqrt(a+b) / 
                    np.diff(df[i]['HeightAboveGroundLevel'])))
    return df


def calculate_and_include_best_fit_features(df, window_size):
    for i in range(len(df)):
        for j in range(len(df[i])):
            if j >= window_size - 1:
                x = df[i]['DeviceCoordinate1'].iloc[j - window_size + 1: j]
                y = df[i]['DeviceCoordinate2'].iloc[j - window_size + 1: j]
                d = df[i]['Distance'].iloc[j - window_size + 1: j].sum()
                line = np.polyfit(x, y, 1)  # Order 1 so straight line is returned
                if d == 0:
                    df[i]['BestFitDisplacementMean'].iloc[j] = (best_fit_diff(line[0], line[1], x, y) * 0).mean()
                    df[i]['BestFitDisplacementVariance'].iloc[j] = (best_fit_diff(line[0], line[1], x, y) * 0).var()
                    df[i]['BestFitDiff'].iloc[j] = (best_fit_diff(line[0], line[1], list(x)[1], list(y)[1]) * 0) - (
                                best_fit_diff(line[0], line[1], list(x)[0], list(y)[0]) * 0)
                else:
                    df[i]['BestFitDisplacementMean'].iloc[j] = (
                                best_fit_diff(line[0], line[1], x, y) * (window_size / d)).mean()
                    df[i]['BestFitDisplacementVariance'].iloc[j] = (
                                best_fit_diff(line[0], line[1], x, y) * (window_size / d)).var()
                    df[i]['BestFitDiff'].iloc[j] = (best_fit_diff(line[0], line[1], list(x)[1], list(y)[1]) * (
                                window_size / d)) - (best_fit_diff(line[0], line[1], list(x)[0], list(y)[0]) * (
                                    window_size / d))
            else:
                df[i]['BestFitDisplacementMean'] = float("nan")
                df[i]['BestFitDisplacementVariance'] = float("nan")
                df[i]['BestFitDiff'] = float("nan")
    return df


def calculate_and_include_slope_diff(df):
    for i in range(len(df)):
        a = abs(np.diff(df[i]['DeviceCoordinate2']))
        b = abs(np.diff(df[i]['DeviceCoordinate1']))
        df[i]['SlopeDiff'] = np.append(np.array(0), np.arctan(np.divide(a, b, out=np.zeros_like(a), where=b != 0)))
    return df


def calculate_and_include_mean_and_var(df, window_size):
    for i in range(len(df)):
        df[i]['HeightMean'] = df[i]['DeviceCoordinate3'].rolling(window=window_size).mean()
        df[i]['HeightVariance'] = df[i]['DeviceCoordinate3'].rolling(window=window_size).var()
#         df[i]['AGLMean'] = df[i]['HeightAboveGroundLevel'].rolling(window=window_size).mean()
#         df[i]['AGLVariance'] = df[i]['HeightAboveGroundLevel'].rolling(window=window_size).var()

#         df[i]['DistanceTotal'] = df[i]['Distance'].rolling(window=window_size).sum()
#         df[i]['DistanceMax'] = df[i]['Distance'].rolling(window=window_size).max()
#         df[i]['DistanceMinMaxRatio'] = df[i]['Distance'].rolling(window=window_size).min()\
#             / df[i]['Distance'].rolling(window=window_size).max()
#         df[i]['RangeMean'] = df[i]['SlantRange'].rolling(window=window_size).mean()
#         df[i]['ScansMean'] = np.append(np.array(0), np.diff(df[i]['ScanNumber']))
        df[i]['VelocityMean'] = df[i]['Velocity'].rolling(window=window_size).mean()
        df[i]['VelocityVariance'] = df[i]['Velocity'].rolling(window=window_size).var()
        df[i]['AccelerationMean'] = df[i]['Acceleration'].rolling(window=window_size).mean()
        df[i]['AccelerationVariance'] = df[i]['Acceleration'].rolling(window=window_size).var()
        df[i]['JerkMean'] = df[i]['Jerk'].rolling(window=window_size).mean()
        df[i]['JerkVariance'] = df[i]['Jerk'].rolling(window=window_size).var()
        df[i]['PowerMean'] = df[i]['Power'].rolling(window=window_size).mean()
        df[i]['PowerVariance'] = df[i]['Power'].rolling(window=window_size).var()
        df[i]['DetectionsMean'] = df[i]['Detections'].rolling(window=window_size).mean()
        df[i]['DetectionsVariance'] = df[i]['Detections'].rolling(window=window_size).var()
        df[i]['VelocityXMean'] = df[i]['ScenarioVelocity1'].rolling(window=window_size).mean()
        df[i]['VelocityXVariance'] = df[i]['ScenarioVelocity1'].rolling(window=window_size).var()
        df[i]['AccelerationXMean'] = df[i]['AccelerationX'].rolling(window=window_size).mean()
        df[i]['AccelerationXVariance'] = df[i]['AccelerationX'].rolling(window=window_size).var()
        df[i]['JerkXMean'] = df[i]['JerkX'].rolling(window=window_size).mean()
        df[i]['JerkXVariance'] = df[i]['JerkX'].rolling(window=window_size).var()
        df[i]['VelocityYMean'] = df[i]['ScenarioVelocity2'].rolling(window=window_size).mean()
        df[i]['VelocityYVariance'] = df[i]['ScenarioVelocity2'].rolling(window=window_size).var()
        df[i]['AccelerationYMean'] = df[i]['AccelerationY'].rolling(window=window_size).mean()
        df[i]['AccelerationYVariance'] = df[i]['AccelerationY'].rolling(window=window_size).var()
        df[i]['JerkYMean'] = df[i]['JerkY'].rolling(window=window_size).mean()
        df[i]['JerkYVariance'] = df[i]['JerkY'].rolling(window=window_size).var() 
        df[i]['VelocityZMean'] = df[i]['VelocityZ'].rolling(window=window_size).mean()
        df[i]['VelocityZVariance'] = df[i]['VelocityZ'].rolling(window=window_size).var()
        df[i]['AccelerationZMean'] = df[i]['AccelerationZ'].rolling(window=window_size).mean()
        df[i]['AccelerationZVariance'] = df[i]['AccelerationZ'].rolling(window=window_size).var() 
        df[i]['JerkZMean'] = df[i]['JerkZ'].rolling(window=window_size).mean()
        df[i]['JerkZVariance'] = df[i]['JerkZ'].rolling(window=window_size).var() 
        df[i]['ThetaMean'] = df[i]['Theta'].rolling(window=window_size).mean()
        df[i]['ThetaVariance'] = df[i]['Theta'].rolling(window=window_size).var()
    return df


# Helpers

def best_fit_diff(slope, intercept, x, y):
    y_fit = slope * x + intercept
    return y - y_fit
