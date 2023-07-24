import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";

const columns = [
  { field: "id", headerName: "Order", width: 50 },
  {
    field: "input",
    headerName: "Input",
    width: 200,
    editable: false,
  },
  {
    field: "output",
    headerName: "Output",
    width: 200,
    editable: false,
  },
  {
    field: "promptTokens",
    headerName: "Prompt Tokens",
    type: "number",
    width: 100,
    editable: false,
  },
  {
    field: "completionTokens",
    headerName: "Completion Tokens",
    width: 100,
  },
  {
    field: "totalCost",
    headerName: "Total Cost",
    width: 100,
  },
  {
    field: "timestamp",
    headerName: "Timestamp",
    width: 100,
    valueGetter: (params) => new Date(params.row.timestamp).toLocaleString(),
  },
  {
    field: "model",
    headerName: "Model",
    width: 100,
  },
  {
    field: "parameters",
    headerName: "Parameters",
    sortable: false,
    width: "auto",
    valueGetter: (params) => JSON.stringify(params.row.parameters),
  },
];

export default function History({
  executions,
  handleExport,
  handleExecutionRowClick,
}) {
  return (
    <div className="history--container">
      <div className="history--top">
        <span className="history--title">Past executions</span>
        <span className="history--export" onClick={handleExport}>
          export
        </span>
      </div>
      <Box sx={{ height: 400, width: "100%" }}>
        <DataGrid
          rows={executions}
          columns={columns}
          onRowClick={handleExecutionRowClick}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 5,
              },
            },
          }}
          pageSizeOptions={[5]}
          disableRowSelectionOnClick
        />
      </Box>
    </div>
  );
}
