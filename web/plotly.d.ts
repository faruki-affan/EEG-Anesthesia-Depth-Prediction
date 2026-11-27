declare module 'plotly.js-dist-min' {
  const Plotly: {
    react(node: HTMLDivElement, data: object[], layout: object, config: object): Promise<unknown>;
    relayout(node: HTMLDivElement, update: object): Promise<unknown>;
    purge(node: HTMLDivElement): void;
  };
  export default Plotly;
}
