function run_crfid_table1_jobs(manifest_file, crfid_root)
% Run the exact Table 1 CRF_ID adapter over a frozen JSON job manifest.
%
% Each job supplies coordinate-only query input and a relational atlas built
% exclusively from the allowed same-fish reference time points. Ground-truth
% endpoint tracking IDs are never loaded by MATLAB inference.

here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(crfid_root, 'Main'));
% The Table 1 adapter was derived from this official alternate-strain run;
% its compare/duplicate helper signatures live here rather than in Main/.
addpath(fullfile(crfid_root, 'Runs', 'Run_MultiCellCalciumImaging', 'CRF'));
addpath(genpath(fullfile(crfid_root, 'UGM')));

jobs = jsondecode(fileread(manifest_file));
if isempty(jobs)
    error('Empty CRF_ID job manifest: %s', manifest_file);
end

for i = 1:numel(jobs)
    job = jobs(i);
    fprintf('[CRF_ID] job %d/%d: %s\n', i, numel(jobs), job.job_id);
    [~, ~, node_label, Neuron_head, conserved_nodeBel] = ...
        annotation_CRF_atanas(job.input_file, 0, 0, 0, 0, job.atlas_file);
    save(job.output_file, 'conserved_nodeBel', 'Neuron_head', 'node_label', '-v7');
end
end
