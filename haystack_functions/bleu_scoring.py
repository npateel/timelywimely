import gzip
from tqdm import tqdm
import json
import sys
import os
from multiprocessing import Pool
import ast
import glob
import pandas as pd
import re
from nltk.translate.bleu_score import sentence_bleu


def dfOverTime(all_outputs, years, topic = True, withGT = False):
    # allOutputs: a list of all the outputs that come from test() for all years
    # years: a list of all the dates in the same order they're put in allOutputs
    #       for the purpose of naming columns
    # topic: boolean choosing whether you include topic columns

    # create empty dataframe all cols
    bleu_cols = [x + " Bleu Score" for x in years[1:]] # don't do bleu for first
    year_ans = [x + " Predicted" for x in years]
    if topic:
        year_topic = [x + " Topic" for x in years]
        year_ans = year_ans + year_topic
    sorted_cols = (year_ans + bleu_cols)
    sorted_cols.sort()
    all_cols = ["Question", "GT Answer", "GT Topic"] + sorted_cols
    df = pd.DataFrame(columns = all_cols)

    # use the first dict as the reference
    base_output = all_outputs[0]

    # create a dictionary to hold lists of all cols
    acc_col_vals = dict.fromkeys(all_cols)
    for key in acc_col_vals:
        acc_col_vals[key] = []
    print(acc_col_vals)
    # populate vals of each col one row at a time
    j = 0
    for q, rest in base_output.items():
        acc_col_vals["Question"].append(q)
        acc_col_vals["GT Answer"].append(rest["gt_ans"])
        acc_col_vals["GT Topic"].append(rest["gt_topic"])
        col_name = years[0] + " Predicted"
        # rest["predicted_ans"] is a list, append a string to the predicted
        # answer col
        ans_str = ""
        for ans in rest["predicted_ans"]:
            ans_str = ans_str + ans + ", "
        acc_col_vals[col_name].append(ans_str[:-2])
        if topic:
            col_top = years[0] + " Topic"
            if len(rest["predicted_ans"]) == 0:
                acc_col_vals[col_top].append("None")
            else:
                acc_col_vals[col_top].append(all_outputs[0][q]["predicted_topic"])

        # add all first year predicted answers to ref for bleu, is this ok?
        ref = [a.split(" ") for a in rest["predicted_ans"]]
        if withGT:
            ref = ref + [a.split(" ") for a in rest["gt_ans"]]
        if len(ref) == 0:
            ref.append([])
        # for the rest of the years add answer and bleu score
        for i in range(1,len(years)):
            year = years[i]
            col_ans = year + " Predicted"
            col_bleu = year + " Bleu Score"
            all_ans = all_outputs[i][q]["predicted_ans"]

            #print(years[i], "ref:", ref)
            #print(all_ans[0].split(" "))
            #print()
            max_bleu = -1
            best_ans = "None"
            # choose the best answer according to unigram bleu
            for ans in all_ans:
                curr_bleu = 100 * sentence_bleu(ref, ans.split(" "), weights=(1,0,0,0))
                if curr_bleu > max_bleu:
                    max_bleu = curr_bleu
                    best_ans = ans
            if topic:
                col_top = year + " Topic"
                if best_ans == "None":
                    acc_col_vals[col_top].append(best_ans)
                else:
                    acc_col_vals[col_top].append(all_outputs[i][q]["predicted_topic"][best_ans])
            #print(best_ans.split(" "), ref)
            acc_col_vals[col_ans].append(best_ans)
            acc_col_vals[col_bleu].append(max_bleu)
            ref.append(best_ans.split(" "))
        if j == 0 or j == 1:
            print("tada")
            print(acc_col_vals)
            j = j + 1

    for col_name, col_data in acc_col_vals.items():
        df[col_name] = col_data

    df['max - min'] = df.filter(regex = r'Bleu Score').max(axis = 1) - df.filter(regex = r'Bleu Score').min(axis = 1)

    return df

def dfGTcompare(all_outputs, years, topic = True):

    bleu_cols = [x + " Bleu Score" for x in years] # don't do bleu for first
    year_ans = [x + " Predicted" for x in years]
    if topic:
        year_topic = [x + " Topic" for x in years]
        year_ans = year_ans + year_topic
    sorted_cols = (year_ans + bleu_cols)
    sorted_cols.sort()
    all_cols = ["Question", "GT Answer", "GT Topic"] + sorted_cols
    df = pd.DataFrame(columns = all_cols)

    base_output = all_outputs[0]

    acc_col_vals = dict.fromkeys(all_cols)
    for key in acc_col_vals:
        acc_col_vals[key] = []
    print("GT", acc_col_vals)
    j = 0
    for q, rest in base_output.items():
        if j == 0:
            print(q, rest)
        acc_col_vals["Question"].append(q)
        acc_col_vals["GT Answer"].append(rest["gt_ans"])
        acc_col_vals["GT Topic"].append(rest["gt_topic"])
        col_name = years[0] + " Predicted"
        # rest["predicted_ans"] is a list, append a string to the predicted
        # answer col
        ans_str = ""
        #for ans in rest["predicted_ans"]:
            #ans_str = ans_str + ans + ", "
        # acc_col_vals[col_name].append(ans_str[:-2])
        ref = [a.split(" ") for a in rest["gt_ans"]]
        if len(ref) == 0:
            ref.append([])
        # for the rest of the years add answer and bleu score
        for i in range(0,len(years)):
            year = years[i]
            col_ans = year + " Predicted"
            col_bleu = year + " Bleu Score"
            all_ans = all_outputs[i][q]["predicted_ans"]

            max_bleu = -1
            best_ans = "None"
            # choose the best answer according to unigram bleu
            for ans in all_ans:
                curr_bleu = 100 * sentence_bleu(ref, ans.split(" "), weights=(1,0,0,0))
                if curr_bleu > max_bleu:
                    max_bleu = curr_bleu
                    best_ans = ans
            if topic:
                col_top = year + " Topic"
                if best_ans == "None":
                    acc_col_vals[col_top].append(best_ans)
                else:
                    acc_col_vals[col_top].append(all_outputs[i][q]["predicted_topic"][best_ans])
            acc_col_vals[col_ans].append(best_ans)
            acc_col_vals[col_bleu].append(max_bleu)
        if j == 0 or j == 1:
            print("here")
            print(acc_col_vals)
            j = j + 1


    for col_name, col_data in acc_col_vals.items():
        df[col_name] = col_data

    df['max - min'] = df.filter(regex = r'Bleu Score').max(axis = 1) - df.filter(regex = r'Bleu Score').min(axis = 1)

    return df


if __name__ == "__main__":
    # get list of os things
    haystacks = os.listdir("../haystack")
    r = re.compile("^20[12][0-9]")
    years = list(filter(r.match, haystacks))
    years.sort()
    dicts = []
    for year in years:
        json_path = os.path.join("../haystack", year)
        json_path = os.path.join(json_path, "predictions.json")
        with open(json_path) as jf:
            dicts.append(json.load(jf))

    df = dfOverTime(dicts, years)
    df.to_csv("../haystack/bleuOvertime.csv")

    df = dfOverTime(dicts, years, withGT = True)
    df.to_csv("../haystack/bleuBoth.csv")

    df = dfGTcompare(dicts, years)
    df.to_csv("../haystack/bleuGT.csv")
