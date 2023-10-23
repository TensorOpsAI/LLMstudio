import { Checkbox } from "../../../components/primitives/Checkbox";

export default function CheckboxParameter({ value, setter, description }) {
  return (
    <div className="items-top flex space-x-2">
      <Checkbox checked={value} onCheckedChange={setter} />
      <div className="grid gap-1.5 leading-none">
        <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
          {description}
        </label>
      </div>
    </div>
  );
}
