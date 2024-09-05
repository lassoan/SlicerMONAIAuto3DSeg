# How to get standard codes for describing segment content

See [this page](https://github.com/lassoan/slicerio/blob/main/UsingStandardTerminology.md) for guidance on how to use standard codes for describing a segmentation.

## Use standard terminology during model training

For model training, you will need to consistently use the same label value for the same segment. However, segmentation (.seg.nrrd) files may have the segments in different order, therefore different label values may be used for the same segment in each file.

You can use the [slicerio Python package](https://pypi.org/project/slicerio/) to get a nrrd file with a consistent `segment name` -> `label value` mapping (see [example](https://github.com/lassoan/slicerio?tab=readme-ov-file#extract-selected-segments-with-chosen-label-values)). This normalization should be done only when preparing data for a specific model training. The normalized .nrrd files do not have to be archived, because they can be regenerated anytime from the source .seg.nrrd files.

## Use standard terminology for model distribution

When your trained a model is ready for distribution with MONAI Auto3DSeg, you need to create a `labels.csv` file that specifies the term for each label value. You will use the same standard codes in this file as you used in the terminology json file above.

The format of this file is standard .csv, with self-describing column names:

```txt
LabelValue,Name,SegmentedPropertyCategoryCodeSequence.CodingSchemeDesignator,SegmentedPropertyCategoryCodeSequence.CodeValue,SegmentedPropertyCategoryCodeSequence.CodeMeaning,SegmentedPropertyTypeCodeSequence.CodingSchemeDesignator,SegmentedPropertyTypeCodeSequence.CodeValue,SegmentedPropertyTypeCodeSequence.CodeMeaning,SegmentedPropertyTypeModifierCodeSequence.CodingSchemeDesignator,SegmentedPropertyTypeModifierCodeSequence.CodeValue,SegmentedPropertyTypeModifierCodeSequence.CodeMeaning,AnatomicRegionSequence.CodingSchemeDesignator,AnatomicRegionSequence.CodeValue,AnatomicRegionSequence.CodeMeaning,AnatomicRegionModifierSequence.CodingSchemeDesignator,AnatomicRegionModifierSequence.CodeValue,AnatomicRegionModifierSequence.CodeMeaning
3,Neoplasm,SCT,49755003,Morphologically Altered Structure,SCT,86049000,"Neoplasm, Primary",,,,SCT,12738006,Brain,,,
2,Edema,SCT,49755003,Morphologically Altered Structure,SCT,79654002,Edema,,,,SCT,12738006,Brain,,,
1,Necrosis,SCT,49755003,Morphologically Altered Structure,SCT,6574001,Necrosis,,,,SCT,12738006,Brain,,,
```

`Name` column contains an internal (project-specific) name for the segment. It is only for convenience during development and testing.
