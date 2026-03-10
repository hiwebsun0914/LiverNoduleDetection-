from monai.utils import GridSampleMode, GridSamplePadMode
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    Resized,
    ToTensord,
)
import numpy as np
import monai
import os
import json
from pathlib import Path


class MRIDataset(monai.data.Dataset):     #train/test
    
    def __init__(self,
                args,
                flag):
      
        self.csv_path = args.csv_path
        self.data_dir = args.data_dir
        self.flag = flag    
        self.annotation = self._load_annotation()
        self.data_dict = self.load_data()

        self.train_transform = Compose(
        [
                LoadImaged(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), image_only=True),
                EnsureChannelFirstd(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"]),
                Resized(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), spatial_size=(128,128,16)),
                RandRotated(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"],range_x=[- np.pi/18, np.pi/18], prob=0.3, mode=GridSampleMode.BILINEAR, padding_mode=GridSamplePadMode.BORDER, keep_size=True),
                RandFlipd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), prob=0.3),     
                NormalizeIntensityd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp")),
                ToTensord(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"))
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), image_only=True),
                EnsureChannelFirstd(keys=["t2","dwi","in_","out_","pre","ap","pvp","dp"]),
                Resized(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"), spatial_size=(128,128,16)),
                NormalizeIntensityd(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp")),
                ToTensord(keys=("t2","dwi","in_","out_","pre","ap","pvp","dp"))
            ])
        

    def _load_annotation(self):
        dataset_root = Path(self.data_dir).resolve().parent
        annotation_path = dataset_root / "labels" / "Annotation.json"
        if not annotation_path.exists():
            return {}
        with annotation_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("Annotation_info", {})

    def _resolve_phase_path(self, patient_id, phase_name, fallback_name):
        # Preferred path for LLD-MMRI public data: UID-based file names from Annotation.json.
        if patient_id in self.annotation:
            for item in self.annotation[patient_id]:
                if item.get("phase") != phase_name:
                    continue
                study_uid = item.get("studyUID", "")
                series_uid = item.get("seriesUID", "")
                phase_dir = Path(self.data_dir) / patient_id / study_uid
                exact = phase_dir / f"{series_uid}.nii.gz"
                if exact.exists():
                    return str(exact)
                candidates = sorted(phase_dir.glob(f"{series_uid}*.nii.gz"))
                if candidates:
                    return str(candidates[0])

        # Fallback path used by older preprocessed private data.
        fallback = Path(self.data_dir) / patient_id / f"{fallback_name}.nii.gz"
        if fallback.exists():
            return str(fallback)

        raise FileNotFoundError(
            f"Cannot resolve file for patient={patient_id}, phase={phase_name}, "
            f"checked Annotation.json mapping and fallback path={fallback}"
        )

 
    def load_data(self):
      
        t2pathlst = []
        dwipathlst = []
        in_pathlst = []
        out_pathlst = []
        prepathlst = []
        apathlst = []
        pvpathlst = []
        dpathlst = []
        self.labellst = []
        phase_list = ['T2WI', 'DWI', 'In Phase', 'Out Phase', 'C-pre', 'C+A', 'C+V', 'C+Delay']

        csvFile=open(self.csv_path,encoding='utf-8-sig')
        lines=csvFile.readlines()   #lines[0] = patient_ID,dataset,AP,pre,PVP,0-1,0-2
        for i in range(1,len(lines)):
            s=lines[i]
            s=s.replace('\n','')
            patient_inf=s.split(',')
            patient_flag = patient_inf[1]


            if patient_flag == str(self.flag):

                patient_id = patient_inf[0]
                t2pathlst.append(self._resolve_phase_path(patient_id, "T2WI", "T2WI"))
                dwipathlst.append(self._resolve_phase_path(patient_id, "DWI", "DWI"))
                in_pathlst.append(self._resolve_phase_path(patient_id, "In Phase", "In Phase"))
                out_pathlst.append(self._resolve_phase_path(patient_id, "Out Phase", "Out Phase"))
                prepathlst.append(self._resolve_phase_path(patient_id, "C-pre", "C-pre"))
                apathlst.append(self._resolve_phase_path(patient_id, "C+A", "C+A"))
                # Keep original code's channel order for compatibility.
                pvpathlst.append(self._resolve_phase_path(patient_id, "C+Delay", "C+Delay"))
                dpathlst.append(self._resolve_phase_path(patient_id, "C+V", "C+V"))

                self.labellst.append(int(patient_inf[2]))



        data_dict = [{'t2': t2path, 'dwi':dwipath, 'in_':in_path, 'out_':out_path, 'pre': prepath, 'ap':apath, 'pvp':pvpath, 'dp':dpath, 'label':label} for t2path, dwipath, in_path, out_path, prepath, apath, pvpath, dpath ,label in 
                        zip(t2pathlst, dwipathlst, in_pathlst, out_pathlst, prepathlst, apathlst, pvpathlst, dpathlst, self.labellst)]
        
        return data_dict
        


    def __getitem__(self,index):
        

        if self.flag == "Train":
            image = self.train_transform(self.data_dict[index])

        else:
            image = self.val_transform(self.data_dict[index])

        label =self.labellst[index]

        return image['t2'], image['dwi'], image['in_'], image['out_'], image['pre'], image['ap'], image['pvp'], image['dp'], label  #self.label = class_label


    def __len__(self):
        return len(self.labellst)
