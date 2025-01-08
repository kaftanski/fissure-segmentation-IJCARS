from data_processing.datasets import LungData
from prepare_fissure_completeness import RESULTS_DIR, COMPLETNESS_DATA_PATH, SEQUENCES
import os
import SimpleITK as sitk
import pandas as pd


def fissure_incompleteness_by_cutoff_value(img, complete_fissure_labels, lower_threshold_hu=-900, upper_threshold_hu=-700):
    non_fissure_mask = sitk.Threshold(img, lower=lower_threshold_hu, upper=upper_threshold_hu, outsideValue=1.)
    non_fissure_mask = sitk.Cast(non_fissure_mask, sitk.sitkUInt8)
    complete_fissure_binary = sitk.BinaryThreshold(complete_fissure_labels, lowerThreshold=1)
    incomplete_fissure_mask = sitk.And(complete_fissure_binary, non_fissure_mask)
    return incomplete_fissure_mask


def incompleteness_per_fissure(complete_fissure_labels, incompleteness_map):
    stats = sitk.LabelStatisticsImageFilter()
    stats.Execute(incompleteness_map, complete_fissure_labels)
    return {label: stats.GetMean(label) for label in stats.GetLabels()}


if __name__ == '__main__':
    image_ds = LungData(COMPLETNESS_DATA_PATH)

    results_df = pd.DataFrame(columns=['case', 'sequence', 'fold', 'fissure', 'incompleteness'])

    for fold in range(5):
        fold_dir = os.path.join(RESULTS_DIR, f'fold{fold}', 'labels')

        # load the predicted complete fissures
        for i, (case, sequence) in enumerate(image_ds.ids):
            predicted_complete_fissure = sitk.ReadImage(os.path.join(fold_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))
            incompleteness_map = fissure_incompleteness_by_cutoff_value(image_ds.get_image(i), predicted_complete_fissure)
            sitk.WriteImage(incompleteness_map, os.path.join(fold_dir, f'{case}_fissures_incompleteness_{sequence}.nii.gz'))

            # compute stats for incompleteness
            print(incompleteness_per_fissure(predicted_complete_fissure, incompleteness_map))

